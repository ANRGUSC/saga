"""Compare Inspirit_FIFO and Inspirit_HEFT across hyperparameter combinations.

Sweeps threshold (dec_step / s_inc / s_dec) and delta_ready for both schedulers
on a single synthetic workflow, asserts correctness, and writes a per-step CSV
to tests/outputs/inspirit_comparison.csv for offline analysis.

Columns mirror big_example_environment/main.py, with an extra 'scheduler' column
to distinguish the two variants.
"""

import csv
import pathlib
from dataclasses import dataclass
from typing import List, Optional

import pytest

from saga import Network, TaskGraph
from saga.schedulers.online import (
    InspiritController,
    InspiritEnvironment,
    ReadyChangeObserver,
    TaskCompletionStep,
    TaskEventStep,
)
from saga.schedulers.online.online_algorithms.FIFO import Inspirit_FIFO_Environment
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    GreedyInsert,
    GreedyInsertCompareFuncs,
    UpwardRanking,
)

SMOOTHING_RATE = 0.8
OUTPUT_DIR = pathlib.Path(__file__).parent / "outputs"

COLUMNS = [
    "scheduler",
    "threshold",
    "delta_ready",
    "heft_makespan",
    "heft_throughput",
    "inspirit_makespan",
    "inspirit_throughput",
    "step",
    "time",
    "state",
    "peak",
    "nready",
    "makespan",
    "dispatched",
    "dispatch_type",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_heft_scheduler() -> ParametricScheduler:
    return ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(
            append_only=False,
            compare=GreedyInsertCompareFuncs.EFT,
            critical_path=False,
        ),
    )


@dataclass
class _StepRecord:
    step: int
    time: float
    state: str
    peak: int
    nready: int
    makespan: float
    dispatched: Optional[str]
    dispatch_type: Optional[str]


def _run_inspirit_fifo(network, task_graph, delta_ready, threshold):
    log: List[_StepRecord] = []

    def on_step(env):
        log.append(_StepRecord(
            step=env._step,
            time=env.current_time,
            state=env.cur_state or "-",
            peak=env.peak,
            nready=len(env.ready_tasks),
            makespan=env.schedule.makespan,
            dispatched=env.last_dispatched,
            dispatch_type=env.last_dispatch_type,
        ))

    env = Inspirit_FIFO_Environment(
        network=network,
        task_graph=task_graph,
        delta_ready=delta_ready,
        on_step=on_step,
        dec_step=threshold,
        s_inc=threshold,
        s_dec=threshold,
    )
    schedule = env.run()
    return schedule, log


def _run_inspirit_heft(network, task_graph, delta_ready, threshold):
    log: List[_StepRecord] = []

    def on_step(env):
        log.append(_StepRecord(
            step=env._step,
            time=env.current_time,
            state=env.cur_state or "-",
            peak=env.peak,
            nready=len(env.ready_tasks),
            makespan=env.schedule.makespan,
            dispatched=env.last_dispatched,
            dispatch_type=env.last_dispatch_type,
        ))

    env = InspiritEnvironment(
        network=network,
        task_graph=task_graph,
        scheduler=make_heft_scheduler(),
        step_strategy=TaskCompletionStep(),
        observer=ReadyChangeObserver(delta_ready),
        time_window=None,
        controller=InspiritController(smoothing_rate=SMOOTHING_RATE),
        on_step=on_step,
        dec_step=threshold,
        s_inc=threshold,
        s_dec=threshold,
    )
    schedule = env.run()
    return schedule, log


def _rows_from_log(
    scheduler_name, threshold, delta_ready,
    heft_makespan, heft_throughput,
    inspirit_makespan, inspirit_throughput,
    log: List[_StepRecord],
):
    return [
        {
            "scheduler": scheduler_name,
            "threshold": threshold,
            "delta_ready": delta_ready,
            "heft_makespan": heft_makespan,
            "heft_throughput": heft_throughput,
            "inspirit_makespan": inspirit_makespan,
            "inspirit_throughput": inspirit_throughput,
            "step": r.step,
            "time": r.time,
            "state": r.state,
            "peak": r.peak,
            "nready": r.nready,
            "makespan": r.makespan,
            "dispatched": r.dispatched,
            "dispatch_type": r.dispatch_type,
        }
        for r in log
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def medium_network():
    return Network.create(
        nodes=[("v1", 2.0), ("v2", 1.5), ("v3", 2.5), ("v4", 1.0)],
        edges=[
            ("v1", "v2", 1.0), ("v1", "v3", 1.0), ("v1", "v4", 1.0),
            ("v2", "v3", 1.0), ("v2", "v4", 1.0), ("v3", "v4", 1.0),
        ],
    )


@pytest.fixture
def medium_task_graph():
    """Fan-out / fan-in DAG: 9 tasks across 4 levels."""
    return TaskGraph.create(
        tasks=[
            ("t1", 2.0), ("t2", 3.0), ("t3", 2.5), ("t4", 1.5),
            ("t5", 2.0), ("t6", 1.8), ("t7", 2.2), ("t8", 1.0), ("t9", 1.5),
        ],
        dependencies=[
            ("t1", "t2", 0.5), ("t1", "t3", 0.5), ("t1", "t4", 0.5),
            ("t2", "t5", 0.3), ("t3", "t5", 0.3), ("t3", "t6", 0.3),
            ("t4", "t6", 0.3), ("t4", "t7", 0.3),
            ("t5", "t8", 0.2), ("t6", "t8", 0.2), ("t7", "t9", 0.2),
            ("t8", "t9", 0.2),
        ],
    )


# ---------------------------------------------------------------------------
# Hyperparameter sweep
# ---------------------------------------------------------------------------

def test_inspirit_fifo_vs_heft_hyperparameter_sweep(medium_network, medium_task_graph):
    """Sweep threshold and delta_ready for Inspirit_FIFO and Inspirit_HEFT.

    For every (threshold, delta_ready) pair:
      - Asserts both schedulers complete and schedule all tasks.
      - Records per-step state alongside final makespan/throughput and
        HEFT baseline metrics.

    Writes results to tests/outputs/inspirit_comparison.csv.
    """
    network = medium_network
    task_graph = medium_task_graph
    workers = len(list(network.nodes))
    all_task_names = {t.name for t in task_graph.tasks}

    heft_schedule = make_heft_scheduler().schedule(network, task_graph)
    heft_makespan = heft_schedule.makespan
    heft_throughput = heft_schedule.throughput

    thresholds = sorted({2, max(1, workers), workers * 2})
    delta_readys = sorted({1, max(1, workers), workers * 2})

    rows = []

    for threshold in thresholds:
        for delta_ready in delta_readys:
            # --- Inspirit-FIFO ---
            fifo_schedule, fifo_log = _run_inspirit_fifo(
                network, task_graph, delta_ready, threshold
            )
            assert fifo_schedule.makespan > 0, (
                f"Inspirit_FIFO makespan <= 0 (threshold={threshold}, delta_ready={delta_ready})"
            )
            fifo_scheduled = {t.name for tasks in fifo_schedule.mapping.values() for t in tasks}
            assert fifo_scheduled == all_task_names, (
                f"Inspirit_FIFO did not schedule all tasks (threshold={threshold}, delta_ready={delta_ready}): "
                f"missing={all_task_names - fifo_scheduled}"
            )
            rows.extend(_rows_from_log(
                "Inspirit_FIFO", threshold, delta_ready,
                heft_makespan, heft_throughput,
                fifo_schedule.makespan, fifo_schedule.throughput,
                fifo_log,
            ))

            # --- Inspirit-HEFT ---
            heft_inspirit_schedule, heft_inspirit_log = _run_inspirit_heft(
                network, task_graph, delta_ready, threshold
            )
            assert heft_inspirit_schedule.makespan > 0, (
                f"Inspirit_HEFT makespan <= 0 (threshold={threshold}, delta_ready={delta_ready})"
            )
            heft_inspirit_scheduled = {
                t.name for tasks in heft_inspirit_schedule.mapping.values() for t in tasks
            }
            assert heft_inspirit_scheduled == all_task_names, (
                f"Inspirit_HEFT did not schedule all tasks (threshold={threshold}, delta_ready={delta_ready}): "
                f"missing={all_task_names - heft_inspirit_scheduled}"
            )
            rows.extend(_rows_from_log(
                "Inspirit_HEFT", threshold, delta_ready,
                heft_makespan, heft_throughput,
                heft_inspirit_schedule.makespan, heft_inspirit_schedule.throughput,
                heft_inspirit_log,
            ))

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_csv = OUTPUT_DIR / "inspirit_comparison.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    assert output_csv.exists()
    assert len(rows) > 0


# ---------------------------------------------------------------------------
# Dispatch confirmation — wide fan-out graph
# ---------------------------------------------------------------------------

@pytest.fixture
def wide_fanout_network():
    """4 workers with full connectivity."""
    return Network.create(
        nodes=[("v1", 2.0), ("v2", 2.0), ("v3", 2.0), ("v4", 2.0)],
        edges=[
            ("v1", "v2", 1.0), ("v1", "v3", 1.0), ("v1", "v4", 1.0),
            ("v2", "v3", 1.0), ("v2", "v4", 1.0), ("v3", "v4", 1.0),
        ],
    )


@pytest.fixture
def wide_fanout_task_graph():
    """Hub task t1 fans out to 5 children, all converging to t7.

    When t1 finishes, ready count jumps from 1 → 5 (delta=4).
    With threshold=1 this guarantees delta > s_inc and the dispatch fires.
    """
    return TaskGraph.create(
        tasks=[
            ("t1", 2.0),
            ("t2", 1.0), ("t3", 1.5), ("t4", 2.0), ("t5", 1.2), ("t6", 0.8),
            ("t7", 1.0),
        ],
        dependencies=[
            ("t1", "t2", 0.0), ("t1", "t3", 0.0), ("t1", "t4", 0.0),
            ("t1", "t5", 0.0), ("t1", "t6", 0.0),
            ("t2", "t7", 0.0), ("t3", "t7", 0.0), ("t4", "t7", 0.0),
            ("t5", "t7", 0.0), ("t6", "t7", 0.0),
        ],
    )


def test_inspirit_fifo_dispatch_fires(wide_fanout_network, wide_fanout_task_graph):
    """Confirm the Inspirit dispatch path executes end-to-end.

    The hub graph guarantees a ready-count jump of 4 when t1 finishes.
    With threshold=1, InspiritController's condition (delta > s_inc) evaluates
    to 4 > 1 = True, so at least one dispatch must fire.

    Checks:
      - At least one step has a non-null dispatched task.
      - dispatch_type is 'efficiency' or 'ability' on every dispatch.
      - Final schedule contains all tasks.
      - Final schedule makespan is positive.
      - After dispatch, FIFO re-fills the rest (all remaining tasks scheduled).
    """
    network = wide_fanout_network
    task_graph = wide_fanout_task_graph
    all_task_names = {t.name for t in task_graph.tasks}

    fifo_schedule, fifo_log = _run_inspirit_fifo(
        network, task_graph, delta_ready=1, threshold=1
    )

    dispatched_steps = [r for r in fifo_log if r.dispatched]
    assert len(dispatched_steps) > 0, (
        "No dispatch fired — InspiritController never selected a priority task. "
        "Check that the ready-count jump exceeds the threshold."
    )
    for r in dispatched_steps:
        assert r.dispatch_type in ("efficiency", "ability"), (
            f"Unexpected dispatch_type={r.dispatch_type!r}"
        )

    scheduled_names = {t.name for tasks in fifo_schedule.mapping.values() for t in tasks}
    assert scheduled_names == all_task_names, (
        f"Missing tasks after dispatch: {all_task_names - scheduled_names}"
    )
    assert fifo_schedule.makespan > 0


def test_inspirit_heft_dispatch_fires(wide_fanout_network, wide_fanout_task_graph):
    """Same dispatch-confirmation test for Inspirit_HEFT as a baseline."""
    network = wide_fanout_network
    task_graph = wide_fanout_task_graph
    all_task_names = {t.name for t in task_graph.tasks}

    heft_schedule, heft_log = _run_inspirit_heft(
        network, task_graph, delta_ready=1, threshold=1
    )

    dispatched_steps = [r for r in heft_log if r.dispatched]
    assert len(dispatched_steps) > 0, (
        "No dispatch fired for Inspirit_HEFT — check threshold vs graph fan-out."
    )
    for r in dispatched_steps:
        assert r.dispatch_type in ("efficiency", "ability"), (
            f"Unexpected dispatch_type={r.dispatch_type!r}"
        )

    scheduled_names = {t.name for tasks in heft_schedule.mapping.values() for t in tasks}
    assert scheduled_names == all_task_names
    assert heft_schedule.makespan > 0
