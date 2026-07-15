import logging
import pathlib
from functools import lru_cache
from typing import Tuple

import networkx as nx

from saga import Network, Schedule, TaskGraph
from saga.schedulers import HeftScheduler
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    GreedyInsert,
    GreedyInsertCompareFuncs,
    UpwardRanking,
)
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

logging.basicConfig(level=logging.INFO)

thisdir = pathlib.Path(__file__).parent.absolute()
savedir = thisdir / "outputs"
savedir.mkdir(exist_ok=True)


@lru_cache(maxsize=None)
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
    task_graph.add_node("t_2", weight=3)
    task_graph.add_node("t_3", weight=2)
    task_graph.add_node("t_4", weight=1)

    task_graph.add_edge("t_1", "t_2", weight=1)
    task_graph.add_edge("t_1", "t_3", weight=1)
    task_graph.add_edge("t_2", "t_4", weight=5)
    task_graph.add_edge("t_3", "t_4", weight=5)

    return Network.from_nx(network), TaskGraph.from_nx(task_graph)


def draw_instance(network: Network, task_graph: TaskGraph):
    logging.basicConfig(level=logging.INFO)
    ax = draw_task_graph(task_graph.graph, use_latex=False)
    fig = ax.get_figure()
    try:
        fig.savefig(str(savedir / "task_graph.png"))  # type: ignore
    except Exception as e:
        logging.error(f"Failed to save task graph figure: {e}")

    ax = draw_network(network.graph, draw_colors=False, use_latex=False)
    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(str(savedir / "network.png"))  # type: ignore


def draw_schedule(schedule: Schedule, name: str, xmax: float | None = None):
    ax = draw_gantt(schedule.mapping, use_latex=False, xmax=xmax)
    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(str(savedir / f"{name}.png"))  # type: ignore


def heft_schedule():
    network, task_graph = get_instance()
    scheduler = HeftScheduler()
    schedule = scheduler.schedule(network, task_graph)
    return schedule


def parametric_schedule():
    network, task_graph = get_instance()
    scheduler = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(
            append_only=False, compare=GreedyInsertCompareFuncs.EFT, critical_path=False
        ),
    )
    schedule = scheduler.schedule(network, task_graph)
    return schedule


def main():
    network, task_graph = get_instance()
    draw_instance(network, task_graph)

    heft_sched = heft_schedule()
    draw_schedule(heft_sched, "heft_gantt")
    parametric_sched = parametric_schedule()
    draw_schedule(parametric_sched, "parametric_gantt")

    max_makespan = max(sched.makespan for sched in [heft_sched, parametric_sched])
    draw_schedule(heft_sched, "heft_gantt_scaled", xmax=max_makespan)
    draw_schedule(parametric_sched, "parametric_gantt_scaled", xmax=max_makespan)

    print(
        f"Makepsans Matched: {heft_sched.makespan} == {parametric_sched.makespan} : {heft_sched.makespan == parametric_sched.makespan}"
    )


if __name__ == "__main__":
    main()
