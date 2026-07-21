import logging
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

import networkx as nx

from saga import Network, Schedule, TaskGraph
from saga.schedulers.online import (
    Environment,
    InspiritPolicy,
    next_completion,
)
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    GreedyInsert,
    GreedyInsertCompareFuncs,
    UpwardRanking,
)
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

logging.basicConfig(level=logging.WARNING)

SMOOTHING_RATE = 0.8

thisdir = pathlib.Path(__file__).parent.absolute()
savedir = thisdir / "outputs"
savedir.mkdir(exist_ok=True)


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
    dec_step: Optional[int] = None,
    s_inc: Optional[int] = None,
    s_dec: Optional[int] = None,
    delta_ready: int = 3,
) -> Tuple[Schedule, Environment, List[InspiritStepRecord]]:
    log: List[InspiritStepRecord] = []

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

    policy = InspiritPolicy(
        smoothing_rate=SMOOTHING_RATE,
        delta_ready=delta_ready,
        dec_step=dec_step,
        s_inc=s_inc,
        s_dec=s_dec,
    )
    env = Environment(
        network=network,
        task_graph=task_graph,
        scheduler=make_heft_scheduler(),
        step=next_completion,
        policy=policy,
        on_step=on_step,
    )
    return env.run(), env, log


def print_inspirit_log(log: List[InspiritStepRecord]) -> None:
    header = (
        f"  {'Step':>4}  {'Time':>8}  {'State':>5}  "
        f"{'Peak':>4}  {'NReady':>6}  {'Makespan':>10}  {'Dispatched':<12}  Type"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rec in log:
        dispatched_str = rec.dispatched or "-"
        type_str = rec.dispatch_type or "-"
        print(
            f"  {rec.step:>4}  {rec.time:>8.4f}  {rec.state:>5}  "
            f"{rec.peak:>4}  {rec.nready:>6}  {rec.makespan:>10.4f}  "
            f"{dispatched_str:<12}  {type_str}"
        )


def save_fig(ax, name: str) -> None:
    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(str(savedir / f"{name}.png"))
        fig.clf()


def main() -> None:
    network, task_graph = get_instance()
    num_tasks = len(list(task_graph.tasks))
    num_workers = len(list(network.nodes))
    threshold = max(1, num_workers)
    delta_ready = max(1, num_workers)

    print(f"Instance: {num_tasks} tasks, {num_workers} workers")

    # Draw instance
    save_fig(draw_task_graph(task_graph.graph, use_latex=False), "task_graph")
    save_fig(draw_network(network.graph, draw_colors=False, use_latex=False), "network")

    # HEFT offline baseline
    heft_schedule = run_heft(network, task_graph)
    print(f"  HEFT (offline)    makespan: {heft_schedule.makespan:.4f}")
    print(f" HEFT (offline) throughput: {heft_schedule.throughput:.4f}")
    save_fig(draw_gantt(heft_schedule.mapping, use_latex=False), "heft_gantt")

    # Inspirit online scheduler
    inspirit_schedule, inspirit_env, inspirit_log = run_inspirit(
        network, task_graph,
        dec_step=threshold,
        s_inc=threshold,
        s_dec=threshold,
        delta_ready=delta_ready,
    )
    print(f"  Inspirit (online) makespan: {inspirit_schedule.makespan:.4f}")
    print(f" Inspirit (online) throughput: {inspirit_schedule.throughput:.4f}")
    save_fig(draw_gantt(inspirit_schedule.mapping, use_latex=False), "inspirit_gantt")

    ctrl = inspirit_env.policy
    n_dispatched = sum(1 for r in inspirit_log if r.dispatched)
    print(
        f"\n  Inspirit: {n_dispatched} dispatch(es) over {len(inspirit_log)} step(s) | "
        f"peak ready: {ctrl.peak} | "
        f"final state: {ctrl.cur_state} | "
        f"k_inc: {ctrl.k_inc:.4f}"
    )
    print("\n  Inspirit step log:")
    print_inspirit_log(inspirit_log)

    xmax = max(heft_schedule.makespan, inspirit_schedule.makespan)
    save_fig(draw_gantt(heft_schedule.mapping, use_latex=False, xmax=xmax), "heft_gantt_scaled")
    save_fig(draw_gantt(inspirit_schedule.mapping, use_latex=False, xmax=xmax), "inspirit_gantt_scaled")


if __name__ == "__main__":
    main()
