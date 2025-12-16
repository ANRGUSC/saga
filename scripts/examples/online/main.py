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
from saga.schedulers.data.wfcommons import get_wfcommons_instance

# ---------------------- Config ----------------------
THISDIR = pathlib.Path(__file__).resolve().parent
FILETYPE = "pdf"  # Change to "pdf" if needed
plt.rcParams["font.size"] = 12
# -----------------------------------------------------


def get_example_instance() -> Tuple[nx.DiGraph, nx.Graph]:
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


    network = nx.Graph()
    network.add_node("1", weight_estimate=1, weight_actual=1)
    network.add_node("2", weight_estimate=1, weight_actual=1)
    network.add_edge("1", "1", weight_estimate=1e9, weight_actual=1e9)
    network.add_edge("2", "2", weight_estimate=1e9, weight_actual=1e9)
    network.add_edge("1", "2", weight_estimate=1, weight_actual=1)
    
    return task_graph, network

def run_example(task_graph: nx.DiGraph,
                network: nx.Graph,
                savedir: pathlib.Path,
                draw_labels: bool = True) -> None:

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
    savedir.mkdir(parents=True, exist_ok=True)
    
    # draw_task_graph(
    #     task_graph_actual,
    #     figsize=(7, 7),
    #     node_weight_offset=10,
    #     draw_edge_weights=draw_labels,
    #     draw_node_weights=draw_labels,
    #     draw_node_labels=draw_labels
    # ).get_figure().savefig(savedir / f"task_graph_actual.{FILETYPE}")
    draw_network(
        network_actual,
        draw_colors=False,
        draw_edge_weights=draw_labels,
        draw_node_weights=draw_labels,
        draw_node_labels=draw_labels
    ).get_figure().savefig(savedir / f"network_actual.{FILETYPE}")
    # draw_task_graph(
    #     task_graph_estimate,
    #     figsize=(7, 7),
    #     node_weight_offset=10,
    #     draw_edge_weights=draw_labels,
    #     draw_node_weights=draw_labels,
    #     draw_node_labels=draw_labels
    # ).get_figure().savefig(savedir / f"task_graph_estimate.{FILETYPE}")
    draw_network(
        network_estimate,
        draw_colors=False,
        draw_edge_weights=draw_labels,
        draw_node_weights=draw_labels,
        draw_node_labels=draw_labels
    ).get_figure().savefig(savedir / f"network_estimate.{FILETYPE}")

    # Draw Gantt Charts
    fig = draw_gantt(sched_offline, xmax=max_makespan, draw_task_labels=draw_labels).get_figure()
    fig.savefig(savedir / f"gantt_offline.{FILETYPE}")
    plt.close(fig)

    fig = draw_gantt(sched_naive_estimate, xmax=max_makespan, draw_task_labels=draw_labels).get_figure()
    fig.savefig(savedir / f"gantt_naive_estimate.{FILETYPE}")
    plt.close(fig)

    fig = draw_gantt(sched_naive_actual, xmax=max_makespan, draw_task_labels=draw_labels).get_figure()
    fig.savefig(savedir / f"gantt_naive_actual.{FILETYPE}")
    plt.close(fig)

    for i, (sched_estimate, sched_hypothetical, sched_actual) in enumerate(zip(scheds_online_estimate, scheds_online_hypothetical, scheds_online_actual)):
        fig = draw_gantt(sched_estimate, xmax=max_makespan, draw_task_labels=draw_labels).get_figure()
        fig.savefig(savedir / f"gantt_online_estimate_{i}.{FILETYPE}")
        plt.close(fig)

        fig = draw_gantt(sched_hypothetical, xmax=max_makespan, draw_task_labels=draw_labels).get_figure()
        fig.savefig(savedir / f"gantt_online_hypothetical_{i}.{FILETYPE}")
        plt.close(fig)

        fig = draw_gantt(sched_actual, xmax=max_makespan, draw_task_labels=draw_labels).get_figure()
        fig.savefig(savedir / f"gantt_online_actual_{i}.{FILETYPE}")
        plt.close(fig)

    # draw final online schedule
    final_sched = scheds_online_actual[-1]
    fig = draw_gantt(final_sched, xmax=max_makespan, draw_task_labels=draw_labels).get_figure()
    fig.savefig(savedir / f"gantt_online_final.{FILETYPE}")
    plt.close(fig)

    # create summary fig with offline, naive estimate, naive actual, and final online schedule
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    axs[0].set_title("Offline Schedule")
    draw_gantt(sched_offline, axis=axs[0], xmax=max_makespan, draw_task_labels=draw_labels)
    axs[1].set_title("Naive Estimate Schedule")
    draw_gantt(sched_naive_estimate, axis=axs[1], xmax=max_makespan, draw_task_labels=draw_labels)
    axs[2].set_title("Naive Actual Schedule")
    draw_gantt(sched_naive_actual, axis=axs[2], xmax=max_makespan, draw_task_labels=draw_labels)
    axs[3].set_title("Final Online Schedule")
    draw_gantt(final_sched, axis=axs[3], xmax=max_makespan, draw_task_labels=draw_labels)

    plt.tight_layout()
    fig.savefig(savedir / f"summary.{FILETYPE}")

def main() -> None:
    # Run manual example
    task_graph_1, network_1 = get_example_instance()
    run_example(task_graph_1, network_1, THISDIR / "example")

    # Run montage example
    network_ep, task_graph_ep = get_wfcommons_instance(
        recipe_name="montage",
        ccr=5,
        max_size_multiplier=1
    )
    savepath = THISDIR / "montage"
    # remove all files in the save directory
    # if savepath.exists():
    #     for file in savepath.iterdir():
    #         file.unlink()
    savepath.mkdir(parents=True, exist_ok=True)
    run_example(task_graph_ep, network_ep, savepath, draw_labels=False)


if __name__ == "__main__":
    main()