import logging
import pathlib
from functools import lru_cache
from typing import Tuple

import networkx as nx

from saga import Network, Schedule, TaskGraph
from saga.schedulers import HeftScheduler, CpopScheduler
from saga.schedulers.throughput.multi_obj import MultiObjScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

logging.basicConfig(level=logging.INFO)

thisdir = pathlib.Path(__file__).parent.absolute()
savedir = thisdir / "outputs"
savedir.mkdir(exist_ok=True)


@lru_cache(maxsize=None)
def get_instance() -> Tuple[Network, TaskGraph]:
    network = nx.Graph()
    network.add_node("v_1", weight=1)
    network.add_node("v_2", weight=1)
    network.add_node("v_3", weight=1)
    network.add_edge("v_1", "v_2", weight=1)
    network.add_edge("v_1", "v_3", weight=1)
    network.add_edge("v_2", "v_3", weight=1)

    task_graph = nx.DiGraph()
    task_graph.add_node("t_1", weight=3)
    task_graph.add_node("t_2", weight=3)
    task_graph.add_node("t_3", weight=3)
    task_graph.add_node("t_4", weight=3)
    task_graph.add_node("t_5", weight=3)

    task_graph.add_edge("t_1", "t_2", weight=2)
    task_graph.add_edge("t_1", "t_3", weight=2)
    task_graph.add_edge("t_1", "t_4", weight=2)
    task_graph.add_edge("t_3", "t_5", weight=3)
    task_graph.add_edge("t_2", "t_5", weight=3)
    task_graph.add_edge("t_4", "t_5", weight=3)

    return Network.from_nx(network), TaskGraph.from_nx(task_graph)


def draw_instance(network: Network, task_graph: TaskGraph):
    ax = draw_task_graph(task_graph.graph, use_latex=False)
    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(str(savedir / "task_graph.png"))

    ax = draw_network(network.graph, draw_colors=False, use_latex=False)
    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(str(savedir / "network.png"))


def draw_schedule(schedule: Schedule, name: str, xmax: float | None = None):
    ax = draw_gantt(schedule.mapping, use_latex=False, xmax=xmax)
    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(str(savedir / f"{name}.png"))


def multi_obj_schedule() -> Schedule:
    network, task_graph = get_instance()
    scheduler = MultiObjScheduler()
    return scheduler.schedule(network, task_graph)


def heft_schedule() -> Schedule:
    network, task_graph = get_instance()
    scheduler = HeftScheduler()
    return scheduler.schedule(network, task_graph)

def cpop_schedule() -> Schedule:
    # for now, just use multi_obj as a placeholder for CPOP
    network, task_graph = get_instance()
    scheduler = CpopScheduler()
    return scheduler.schedule(network, task_graph)


def main():
    draw_instance(*get_instance())

    multi_obj_sched = multi_obj_schedule()
    heft_sched = heft_schedule()
    cpop_sched = cpop_schedule()  # for now, just use multi_obj as a placeholder for CPOP

    max_makespan = max(multi_obj_sched.makespan, heft_sched.makespan, cpop_sched.makespan)
    max_throughput = max(multi_obj_sched.throughput, heft_sched.throughput)
    print(f"MultiObj makespan: {multi_obj_sched.makespan}")
    print(f"MultiObj throughput: {multi_obj_sched.throughput}")


    print(f"HEFT makespan:     {heft_sched.makespan}")
    print(f"HEFT throughput:     {heft_sched.throughput}")

    print(f"CPOP makespan:     {cpop_sched.makespan}")
    print(f"CPOP throughput:     {cpop_sched.throughput}")

    draw_schedule(multi_obj_sched, "multi_obj_gantt")
    draw_schedule(heft_sched, "heft_gantt")
    draw_schedule(multi_obj_sched, "multi_obj_gantt_scaled", xmax=max_makespan)
    draw_schedule(heft_sched, "heft_gantt_scaled", xmax=max_makespan)
    draw_schedule(cpop_sched, "cpop_gantt")
    draw_schedule(cpop_sched, "cpop_gantt_scaled", xmax=max_makespan)


if __name__ == "__main__":
    main()
