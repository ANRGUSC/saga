import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import networkx as nx

from saga import Network, Schedule, TaskGraph
from saga.schedulers.online import (
    Environment,
    FrontierEnvironment,
    InspiritPolicy,
    FrontierFillPolicy,
    next_event,
    next_completion,
)
from saga.schedulers.online.algorithms.fifo import FIFOEnvironment
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    GreedyInsert,
    GreedyInsertCompareFuncs,
    UpwardRanking,
)

SMOOTHING_RATE = 0.8

thisdir = pathlib.Path(__file__).parent.absolute()


def get_instance() -> Tuple[Network, TaskGraph]:
    network = nx.Graph()
    network.add_node("v_1", weight=1)
    network.add_node("v_2", weight=2)
    network.add_node("v_3", weight=2)
    network.add_edge("v_1", "v_2", weight=2)
    network.add_edge("v_1", "v_3", weight=2)
    network.add_edge("v_2", "v_3", weight=1)

    task_graph = nx.DiGraph()
    task_graph.add_node("t_1", weight=1)
    task_graph.add_node("t_2", weight=8)
    task_graph.add_node("t_3", weight=8)
    task_graph.add_node("t_4", weight=8)
    task_graph.add_node("t_5", weight=5)
    task_graph.add_node("t_6", weight=1)
    task_graph.add_node("t_7", weight=5)
    task_graph.add_node("t_8", weight=2)
    task_graph.add_node("t_9", weight=2)
    task_graph.add_node("t_10", weight=2)
    task_graph.add_node("t_11", weight=2)
    task_graph.add_node("t_12", weight=2)
    task_graph.add_node("t_13", weight=2)
    task_graph.add_node("t_14", weight=2)
    task_graph.add_node("t_15", weight=2)
    task_graph.add_node("t_16", weight=2)
    task_graph.add_node("t_17", weight=2)
    task_graph.add_node("t_18", weight=6)
    task_graph.add_node("t_19", weight=5)
    task_graph.add_node("t_20", weight=5)
    task_graph.add_node("t_21", weight=1)
    task_graph.add_node("t_22", weight=1)
    task_graph.add_node("t_23", weight=1)
    task_graph.add_node("t_24", weight=4)
    task_graph.add_node("t_25", weight=4)
    task_graph.add_node("t_26", weight=4)
    task_graph.add_node("t_27", weight=2)
    task_graph.add_node("t_28", weight=2)
    task_graph.add_node("t_29", weight=2)
    task_graph.add_node("t_30", weight=2)
    task_graph.add_node("t_31", weight=2)
    task_graph.add_node("t_32", weight=1)

    task_graph.add_edge("t_1", "t_2", weight=2)
    task_graph.add_edge("t_1", "t_3", weight=2)
    task_graph.add_edge("t_1", "t_4", weight=2)
    task_graph.add_edge("t_2", "t_9", weight=2)
    task_graph.add_edge("t_2", "t_10", weight=2)
    task_graph.add_edge("t_2", "t_11", weight=2)
    task_graph.add_edge("t_3", "t_18", weight=2)
    task_graph.add_edge("t_4", "t_5", weight=2)
    task_graph.add_edge("t_5", "t_6", weight=2)
    task_graph.add_edge("t_5", "t_7", weight=2)
    task_graph.add_edge("t_5", "t_8", weight=2)
    task_graph.add_edge("t_6", "t_24", weight=2)
    task_graph.add_edge("t_6", "t_25", weight=2)
    task_graph.add_edge("t_6", "t_26", weight=2)
    task_graph.add_edge("t_7", "t_27", weight=2)
    task_graph.add_edge("t_8", "t_27", weight=2)
    task_graph.add_edge("t_9", "t_12", weight=2)
    task_graph.add_edge("t_9", "t_13", weight=2)
    task_graph.add_edge("t_10", "t_14", weight=2)
    task_graph.add_edge("t_10", "t_15", weight=2)
    task_graph.add_edge("t_11", "t_16", weight=2)
    task_graph.add_edge("t_11", "t_17", weight=2)
    task_graph.add_edge("t_12", "t_21", weight=2)
    task_graph.add_edge("t_13", "t_21", weight=2)
    task_graph.add_edge("t_14", "t_22", weight=2)
    task_graph.add_edge("t_15", "t_22", weight=2)
    task_graph.add_edge("t_16", "t_23", weight=2)
    task_graph.add_edge("t_17", "t_23", weight=2)
    task_graph.add_edge("t_18", "t_19", weight=2)
    task_graph.add_edge("t_18", "t_20", weight=2)
    task_graph.add_edge("t_19", "t_32", weight=2)
    task_graph.add_edge("t_20", "t_32", weight=2)
    task_graph.add_edge("t_21", "t_32", weight=2)
    task_graph.add_edge("t_22", "t_32", weight=2)
    task_graph.add_edge("t_23", "t_32", weight=2)
    task_graph.add_edge("t_24", "t_32", weight=2)
    task_graph.add_edge("t_25", "t_32", weight=2)
    task_graph.add_edge("t_26", "t_32", weight=2)
    task_graph.add_edge("t_27", "t_28", weight=2)
    task_graph.add_edge("t_27", "t_29", weight=2)
    task_graph.add_edge("t_27", "t_30", weight=2)
    task_graph.add_edge("t_27", "t_31", weight=2)
    task_graph.add_edge("t_28", "t_32", weight=2)
    task_graph.add_edge("t_29", "t_32", weight=2)
    task_graph.add_edge("t_30", "t_32", weight=2)
    task_graph.add_edge("t_31", "t_32", weight=2)

    return Network.from_nx(network), TaskGraph.from_nx(task_graph)


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
# Step record helpers
# ---------------------------------------------------------------------------

@dataclass
class FIFOStepRecord:
    step: int
    time: float
    nready: int
    frontier_size: int
    makespan: float


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


# ---------------------------------------------------------------------------
# FIFO
# ---------------------------------------------------------------------------

def run_fifo(network: Network, task_graph: TaskGraph) -> Tuple[Schedule, List[FIFOStepRecord]]:
    log: List[FIFOStepRecord] = []

    def on_step(env: FrontierEnvironment) -> None:
        log.append(FIFOStepRecord(
            step=env._step,
            time=env.current_time,
            nready=len(env.ready_tasks),
            frontier_size=len(env.frontier),
            makespan=env.schedule.makespan,
        ))

    env = FIFOEnvironment(network=network, task_graph=task_graph, on_step=on_step)
    return env.run(), log


def print_fifo_log(log: List[FIFOStepRecord]) -> None:
    header = f"  {'Step':>4}  {'Time':>8}  {'NReady':>6}  {'Frontier':>8}  {'Makespan':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rec in log:
        print(
            f"  {rec.step:>4}  {rec.time:>8.4f}  {rec.nready:>6}  "
            f"{rec.frontier_size:>8}  {rec.makespan:>10.4f}"
        )


# ---------------------------------------------------------------------------
# Inspirit (shared log/print for FIFO-Inspirit and HEFT-Inspirit)
# ---------------------------------------------------------------------------

def _inspirit_on_step(log: List[InspiritStepRecord]):
    def on_step(env: Environment) -> None:
        ctrl = env.policy
        assert isinstance(ctrl, InspiritPolicy)
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
    return on_step


def print_inspirit_log(log: List[InspiritStepRecord]) -> None:
    header = (
        f"  {'Step':>4}  {'Time':>8}  {'State':>5}  "
        f"{'Peak':>4}  {'NReady':>6}  {'Makespan':>10}  {'Dispatched':<12}  Type"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rec in log:
        print(
            f"  {rec.step:>4}  {rec.time:>8.4f}  {rec.state:>5}  "
            f"{rec.peak:>4}  {rec.nready:>6}  {rec.makespan:>10.4f}  "
            f"{rec.dispatched or '-':<12}  {rec.dispatch_type or '-'}"
        )


# ---------------------------------------------------------------------------
# FIFO-Inspirit
# ---------------------------------------------------------------------------

def run_fifo_inspirit(
    network: Network,
    task_graph: TaskGraph,
    threshold: int,
    delta_ready: int,
) -> Tuple[Schedule, List[InspiritStepRecord]]:
    log: List[InspiritStepRecord] = []

    env = FrontierEnvironment(
        network=network,
        task_graph=task_graph,
        step=next_event,
        policy=InspiritPolicy(
            smoothing_rate=SMOOTHING_RATE,
            delta_ready=delta_ready,
            dec_step=threshold,
            s_inc=threshold,
            s_dec=threshold,
            fill_policy=FrontierFillPolicy(),
        ),
        on_step=_inspirit_on_step(log),
    )
    return env.run(), log


# ---------------------------------------------------------------------------
# HEFT-Inspirit
# ---------------------------------------------------------------------------

def run_heft_inspirit(
    network: Network,
    task_graph: TaskGraph,
    threshold: int,
    delta_ready: int,
) -> Tuple[Schedule, List[InspiritStepRecord]]:
    log: List[InspiritStepRecord] = []

    env = Environment(
        network=network,
        task_graph=task_graph,
        scheduler=make_heft_scheduler(),
        step=next_completion,
        policy=InspiritPolicy(
            smoothing_rate=SMOOTHING_RATE,
            delta_ready=delta_ready,
            dec_step=threshold,
            s_inc=threshold,
            s_dec=threshold,
        ),
        on_step=_inspirit_on_step(log),
    )
    return env.run(), log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    network, task_graph = get_instance()
    num_tasks = len(list(task_graph.tasks))
    num_workers = len(list(network.nodes))
    threshold = max(1, num_workers)
    delta_ready = max(1, num_workers)

    print(f"Instance: {num_tasks} tasks, {num_workers} workers")
    print(f"Threshold / delta_ready: {threshold}\n")

    # --- FIFO ---
    fifo_schedule, fifo_log = run_fifo(network, task_graph)
    print(f"FIFO  makespan: {fifo_schedule.makespan:.4f}  throughput: {fifo_schedule.throughput:.4f}")
    print(f"  {len(fifo_log)} steps\n")
    print_fifo_log(fifo_log)
    print()

    # --- FIFO Inspirit ---
    fifo_ins_schedule, fifo_ins_log = run_fifo_inspirit(network, task_graph, threshold, delta_ready)
    fifo_ins_ctrl = None  # controller state available via log
    n_dispatched = sum(1 for r in fifo_ins_log if r.dispatched)
    print(f"FIFO-Inspirit  makespan: {fifo_ins_schedule.makespan:.4f}  throughput: {fifo_ins_schedule.throughput:.4f}")
    print(f"  {len(fifo_ins_log)} steps, {n_dispatched} dispatch(es)\n")
    print_inspirit_log(fifo_ins_log)
    print()

    # --- HEFT Inspirit ---
    heft_ins_schedule, heft_ins_log = run_heft_inspirit(network, task_graph, threshold, delta_ready)
    n_dispatched = sum(1 for r in heft_ins_log if r.dispatched)
    print(f"HEFT-Inspirit  makespan: {heft_ins_schedule.makespan:.4f}  throughput: {heft_ins_schedule.throughput:.4f}")
    print(f"  {len(heft_ins_log)} steps, {n_dispatched} dispatch(es)\n")
    print_inspirit_log(heft_ins_log)
    print()


if __name__ == "__main__":
    main()
