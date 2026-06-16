"""
big_example_environment/main.py

Sweeps Inspirit threshold and delta_ready parameters across wfcommons recipes,
recording per-step state to a CSV for offline analysis. HEFT makespan/throughput
is included in the CSV as a baseline for each recipe instance.
"""

import logging
import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from saga import Network, Schedule, TaskGraph
from saga.schedulers.data.wfcommons import get_networks, get_workflows
from saga.schedulers.online import (
    Environment,
    InspiritController,
    TaskCompletionStep,
    ReadyChangeObserver,
)
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    GreedyInsert,
    GreedyInsertCompareFuncs,
    UpwardRanking,
)

logging.basicConfig(level=logging.WARNING)

RECIPES = ["montage", "epigenomics", "cycles", "seismology"]
CCR = 1.0
SMOOTHING_RATE = 0.8

thisdir = pathlib.Path(__file__).parent.absolute()
outdir = thisdir / "outputs"
outdir.mkdir(exist_ok=True)


COLUMNS = [
    "recipe",
    "num_tasks",
    "num_workers",
    "heft_makespan",
    "heft_throughput",
    "threshold",
    "delta_ready",
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
# Shared scheduler factory
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


# ---------------------------------------------------------------------------
# Instance generation via wfcommons
# ---------------------------------------------------------------------------

def get_instance(recipe_name: str, ccr: float = CCR) -> Tuple[Network, TaskGraph]:
    networks = get_networks(num=1, cloud_name="chameleon")
    workflows = get_workflows(num=1, recipe_name=recipe_name)
    network = networks[0].scale_to_ccr(workflows[0], ccr)
    return network, workflows[0]


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------

def run_heft(network: Network, task_graph: TaskGraph) -> Schedule:
    return make_heft_scheduler().schedule(network, task_graph)


@dataclass
class InspiritStepRecord:
    step: int
    time: float
    state: str
    peak: int
    nready: int
    makespan: float
    dispatched: Optional[str]
    dispatch_type: Optional[str]


def run_inspirit(
    network: Network,
    task_graph: TaskGraph,
    num_tasks: int,
    delta_ready: int,
    dec_step: Optional[int] = None,
    s_inc: Optional[int] = None,
    s_dec: Optional[int] = None,
) -> Tuple[Schedule, Environment, List[InspiritStepRecord]]:
    log: List[InspiritStepRecord] = []
    pbar = tqdm(
        total=num_tasks,
        desc=f"  threshold={s_inc} delta_ready={delta_ready}",
        unit="task",
        leave=False,
    )
    finished_count = 0

    def on_step(env: Environment) -> None:
        nonlocal finished_count
        ctrl = env.controller
        assert isinstance(ctrl, InspiritController)
        log.append(InspiritStepRecord(
            step=env._step,
            time=env.current_time,
            state=ctrl.cur_state or "-",
            peak=ctrl.peak,
            nready=len(env.ready_tasks),
            makespan=env.schedule.makespan,
            dispatched=ctrl.last_dispatched,
            dispatch_type=ctrl.last_dispatch_type,
        ))
        new_finished = len(env.finished_tasks)
        pbar.update(new_finished - finished_count)
        finished_count = new_finished

    controller = InspiritController(
        smoothing_rate=SMOOTHING_RATE,
        dec_step=dec_step,
        s_inc=s_inc,
        s_dec=s_dec,
    )
    env = Environment(
        network=network,
        task_graph=task_graph,
        scheduler=make_heft_scheduler(),
        step_strategy=TaskCompletionStep(),
        observer=ReadyChangeObserver(delta_ready),
        controller=controller,
        on_step=on_step,
    )
    result = env.run(), env, log
    pbar.close()
    return result


# ---------------------------------------------------------------------------
# Per-recipe runner
# ---------------------------------------------------------------------------

def run_recipe(recipe_name: str, output_csv: pathlib.Path) -> None:
    """Run the full parameter sweep for one recipe, appending rows to output_csv."""
    print(f"\n{'=' * 60}")
    print(f"Recipe: {recipe_name}")
    print(f"{'=' * 60}")

    network, task_graph = get_instance(recipe_name, CCR)
    num_tasks = len(list(task_graph.tasks))
    num_workers = len(list(network.nodes))
    print(f"  Tasks: {num_tasks}, Workers: {num_workers}")

    # HEFT baseline first so it's available for comparison
    heft_schedule = run_heft(network, task_graph)
    heft_makespan = heft_schedule.makespan
    heft_throughput = heft_schedule.throughput
    print(f"  HEFT (offline)    makespan: {heft_makespan:.8f}, throughput: {heft_throughput:.8f}")

    thresholds = sorted({2, max(1, num_workers), num_workers * 2, num_workers * 3, num_workers * 4})
    delta_readys = sorted({2, max(1, num_workers), num_workers * 2, num_workers * 3, num_workers * 4})

    for threshold in thresholds:
        for delta_ready in delta_readys:
            inspirit_schedule, inspirit_env, inspirit_log = run_inspirit(
                network=network,
                task_graph=task_graph,
                num_tasks=num_tasks,
                delta_ready=delta_ready,
                dec_step=threshold,
                s_inc=threshold,
                s_dec=threshold,
            )
            n_dispatched = sum(1 for r in inspirit_log if r.dispatched)
            print(
                f"  threshold={threshold} delta_ready={delta_ready}: "
                f"makespan={inspirit_schedule.makespan:.8f} "
                f"throughput={inspirit_schedule.throughput:.8f} "
                f"dispatches={n_dispatched}"
            )

            rows = [
                {
                    "recipe": recipe_name,
                    "num_tasks": num_tasks,
                    "num_workers": num_workers,
                    "heft_makespan": heft_makespan,
                    "heft_throughput": heft_throughput,
                    "threshold": threshold,
                    "delta_ready": delta_ready,
                    "inspirit_makespan": inspirit_schedule.makespan,
                    "inspirit_throughput": inspirit_schedule.throughput,
                    "step": entry.step,
                    "time": entry.time,
                    "state": entry.state,
                    "peak": entry.peak,
                    "nready": entry.nready,
                    "makespan": entry.makespan,
                    "dispatched": entry.dispatched,
                    "dispatch_type": entry.dispatch_type,
                }
                for entry in inspirit_log
            ]
            pd.DataFrame(rows, columns=COLUMNS).to_csv(
                output_csv, mode="a", header=False, index=False
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_csv = outdir / "output_data.csv"
    # Write header once, clearing any previous run
    pd.DataFrame(columns=COLUMNS).to_csv(output_csv, index=False)
    print(f"Output: {output_csv}")

    for recipe in RECIPES:
        try:
            run_recipe(recipe, output_csv)
        except Exception as e:
            import traceback
            print(f"\n  [{recipe}] Failed: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
