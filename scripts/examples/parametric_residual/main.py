from matplotlib import pyplot as plt
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import GreedyInsert, UpwardRanking, ScheduleType

from saga.utils.random_graphs import get_network, get_branching_dag, add_ccr_weights, add_random_weights
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

import pathlib
import networkx as nx

thisdir = pathlib.Path(__file__).resolve().parent

def main():
    scheduler = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(
            append_only=False,
            compare="EFT",
            critical_path=False
        )
    )

    arrival_times = [2, 4, 5]
    task_graphs = [
        add_random_weights(get_branching_dag(levels=2, branching_factor=2))
        for _ in arrival_times
    ]

    # Rename nodes with prefixes t1_, t2_, etc.
    for i, _tg in enumerate(task_graphs, start=1):  # start=1 to match t1, t2, etc.
        mapping = {node: f"{i}.{node}" for node in _tg.nodes}
        nx.relabel_nodes(_tg, mapping, copy=False)  # Modify in-place

    network = add_ccr_weights(task_graphs[0], add_random_weights(get_network()), ccr=1.0)

    schedule: ScheduleType = {node: [] for node in network.nodes}
    for i, (arrival_time, task_graph) in enumerate(zip(arrival_times, task_graphs), start=1):
        schedule = scheduler.schedule(
            network=network,
            task_graph=task_graph,
            schedule=schedule,
            min_start_time=arrival_time
        )

        ax_task_graph: plt.Axes = draw_task_graph(task_graph)
        ax_task_graph.get_figure().savefig(thisdir / f"task_graph_{i}.png")
        plt.close(ax_task_graph.get_figure())
        
        ax_schedule: plt.Axes = draw_gantt(schedule)
        ax_schedule.get_figure().savefig(thisdir / f"schedule_{i}.png")
        plt.close(ax_schedule.get_figure())

    ax_network: plt.Axes = draw_network(network)
    ax_network.get_figure().savefig(thisdir / "network.png")
    plt.close(ax_network.get_figure())


if __name__ == '__main__':
    main()
