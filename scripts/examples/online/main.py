"""
Online vs Offline Scheduling Example

This script demonstrates how online scheduling works compared to offline scheduling.
It creates a simple 5-task workflow and shows the difference between:
- Offline scheduling (perfect knowledge)
- Naive online scheduling (use estimates, convert to actual)
- True online scheduling (iteratively adapt to actual execution times)
"""


import pathlib
from typing import Callable, Dict, List, Tuple, Union, TypeVar
from multiprocessing import Value, Lock

from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.online_parametric import OnlineParametricScheduler
from saga.schedulers.parametric.components import (
    CPoPRanking, UpwardRanking, GreedyInsert
)
import networkx as nx
import matplotlib.pyplot as plt

from saga.utils.online_tools import schedule_estimate_to_actual, set_weights
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

# ---------------------- Config ----------------------
THISDIR = pathlib.Path(__file__).resolve().parent
SAVEPATH = THISDIR / "example"


def get_task_graph() -> nx.DiGraph:
    task_graph = nx.DiGraph()
    task_graph.add_node("A", weight_estimate=2, weight_actual=2)
    task_graph.add_node("B", weight_estimate=5, weight_actual=1)
    task_graph.add_node("C", weight_estimate=3, weight_actual=3)
    task_graph.add_node("D", weight_estimate=3, weight_actual=5)
    task_graph.add_node("E", weight_estimate=1, weight_actual=1)

    task_graph.add_edge("A", "B", weight_estimate=1, weight_actual=1)
    task_graph.add_edge("A", "C", weight_estimate=1, weight_actual=1)
    task_graph.add_edge("A", "D", weight_estimate=1, weight_actual=1)
    task_graph.add_edge("B", "E", weight_estimate=1, weight_actual=1)
    task_graph.add_edge("C", "E", weight_estimate=1, weight_actual=1)
    task_graph.add_edge("D", "E", weight_estimate=1, weight_actual=1)


    return task_graph

def get_network() -> nx.Graph:
    network = nx.Graph()
    network.add_node("1", weight_estimate=1, weight_actual=1)
    network.add_node("2", weight_estimate=1, weight_actual=1)
    network.add_edge("1", "1", weight_estimate=1e9, weight_actual=1e9)
    network.add_edge("2", "2", weight_estimate=1e9, weight_actual=1e9)
    network.add_edge("1", "2", weight_estimate=1, weight_actual=1)
    return network

def main():
    task_graph = get_task_graph()
    network = get_network()

    scheduler_offline = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(
            append_only=False,
            compare="EFT",
            critical_path=False
        )
    )
    scheduler_online = OnlineParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(
            append_only=False,
            compare="EFT",
            critical_path=False
        )
    )

    task_graph_actual = set_weights(task_graph, "weight_actual")
    network_actual = set_weights(network, "weight_actual")
    task_graph_estimate = set_weights(task_graph, "weight_estimate")
    network_estimate = set_weights(network, "weight_estimate")

    sched_offline = scheduler_offline.schedule(
        network=network_actual,
        task_graph=task_graph_actual
    )
    sched_naive_estimate = scheduler_offline.schedule(
        network=network_estimate,
        task_graph=task_graph_estimate
    )
    sched_naive_actual = schedule_estimate_to_actual(
        network=network_estimate,
        task_graph=task_graph_estimate,
        schedule_estimate=sched_naive_estimate
    )

    scheds_online_estimate, scheds_online_hypothetical, scheds_online_actual = scheduler_online.schedule_iterative(
        network=network_estimate,
        task_graph=task_graph_estimate
    )

    all_schedules = [
        sched_offline, sched_naive_estimate, sched_naive_actual,
        *scheds_online_estimate, *scheds_online_hypothetical, *scheds_online_actual
    ]
    max_makespan = max(
        task.end for sched in all_schedules
        for tasks in sched.values() for task in tasks
    )

    # Draw Task Graph and Network
    SAVEPATH.mkdir(parents=True, exist_ok=True)
    draw_task_graph(task_graph_actual, figsize=(7, 7), node_weight_offset=10).get_figure().savefig(SAVEPATH / "task_graph_actual.png")
    draw_network(network_actual, draw_colors=False).get_figure().savefig(SAVEPATH / "network_actual.png")
    draw_task_graph(task_graph_estimate, figsize=(7, 7), node_weight_offset=10).get_figure().savefig(SAVEPATH / "task_graph_estimate.png")
    draw_network(network_estimate, draw_colors=False).get_figure().savefig(SAVEPATH / "network_estimate.png")

    # Draw Gantt Charts
    fig = draw_gantt(sched_offline, xmax=max_makespan).get_figure()
    fig.savefig(SAVEPATH / "gantt_offline.png")
    plt.close(fig)

    fig = draw_gantt(sched_naive_estimate, xmax=max_makespan).get_figure()
    fig.savefig(SAVEPATH / "gantt_naive_estimate.png")
    plt.close(fig)

    fig = draw_gantt(sched_naive_actual, xmax=max_makespan).get_figure()
    fig.savefig(SAVEPATH / "gantt_naive_actual.png")
    plt.close(fig)

    for i, (sched_estimate, sched_hypothetical, sched_actual) in enumerate(zip(scheds_online_estimate, scheds_online_hypothetical, scheds_online_actual)):
        fig = draw_gantt(sched_estimate, xmax=max_makespan).get_figure()
        fig.savefig(SAVEPATH / f"gantt_online_estimate_{i}.png")
        plt.close(fig)

        fig = draw_gantt(sched_hypothetical, xmax=max_makespan).get_figure()
        fig.savefig(SAVEPATH / f"gantt_online_hypothetical_{i}.png")
        plt.close(fig)

        fig = draw_gantt(sched_actual, xmax=max_makespan).get_figure()
        fig.savefig(SAVEPATH / f"gantt_online_actual_{i}.png")
        plt.close(fig)

if __name__ == "__main__":
    main()