import itertools
from typing import Callable, Dict, Tuple
from saga.common.heft import HeftScheduler
from saga.common.brute_force import BruteForceScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from plotly.graph_objects import Figure 
import pathlib

thisdir = pathlib.Path(__file__).parent.absolute()

def get_example_one() -> Tuple[nx.DiGraph, nx.Graph]:
    task_graph = nx.DiGraph()
    task_graph.add_node("src", weight=1e-9)
    task_graph.add_node("dst", weight=1e-9)

    task_graph.add_node("A", weight=2)
    task_graph.add_node("B", weight=2)
    task_graph.add_node("C", weight=1)
    task_graph.add_node("D", weight=1)

    task_graph.add_edge("A", "C", weight=5)
    task_graph.add_edge("A", "D", weight=1)
    task_graph.add_edge("B", "C", weight=1)
    task_graph.add_edge("B", "D", weight=5)

    for node in task_graph.nodes:
        if task_graph.in_degree(node) == 0 and node not in ["src", "dst"]:
            task_graph.add_edge("src", node, weight=1e-9)
        if task_graph.out_degree(node) == 0 and node not in ["src", "dst"]:
            task_graph.add_edge(node, "dst", weight=1e-9)

    network = nx.Graph()
    network.add_node(0, weight=2)
    network.add_node(1, weight=2)
    network.add_node(2, weight=1)
    network.add_edge(0, 1, weight=1)
    network.add_edge(0, 2, weight=1)
    network.add_edge(1, 2, weight=2)

    for node in network.nodes:
        network.add_edge(node, node, weight=1e9)

    return task_graph, network

def get_example_two() -> Tuple[nx.DiGraph, nx.Graph]:
    task_graph = nx.DiGraph()
    task_graph.add_node("src", weight=1e-9)
    task_graph.add_node("dst", weight=1e-9)

    task_graph.add_node("A", weight=1)
    task_graph.add_node("B", weight=2)
    task_graph.add_node("C", weight=3)
    task_graph.add_node("D", weight=4)

    task_graph.add_edge("A", "C", weight=1)
    task_graph.add_edge("A", "D", weight=2)
    task_graph.add_edge("B", "C", weight=2)
    task_graph.add_edge("B", "D", weight=1)

    for node in task_graph.nodes:
        if task_graph.in_degree(node) == 0 and node not in ["src", "dst"]:
            task_graph.add_edge("src", node, weight=1e-9)
        if task_graph.out_degree(node) == 0 and node not in ["src", "dst"]:
            task_graph.add_edge(node, "dst", weight=1e-9)

    network = nx.Graph()
    network.add_node(0, weight=3)
    network.add_node(1, weight=2)
    network.add_node(2, weight=1)
    network.add_edge(0, 1, weight=5)
    network.add_edge(0, 2, weight=10)
    network.add_edge(1, 2, weight=2)

    for node in network.nodes:
        network.add_edge(node, node, weight=1e9)

    return task_graph, network
    
    
def get_example_three() -> Tuple[nx.DiGraph, nx.Graph]: # bad example
    task_graph = nx.DiGraph()
    task_graph.add_node("src", weight=1e-9)
    task_graph.add_node("dst", weight=1e-9)

    task_graph.add_node("A", weight=1)
    task_graph.add_node("B", weight=1)
    task_graph.add_node("C", weight=2)
    task_graph.add_node("D", weight=2)

    task_graph.add_edge("A", "C", weight=2)
    task_graph.add_edge("A", "D", weight=2)
    task_graph.add_edge("B", "C", weight=2)
    task_graph.add_edge("B", "D", weight=2)

    for node in task_graph.nodes:
        if task_graph.in_degree(node) == 0 and node not in ["src", "dst"]:
            task_graph.add_edge("src", node, weight=1e-9)
        if task_graph.out_degree(node) == 0 and node not in ["src", "dst"]:
            task_graph.add_edge(node, "dst", weight=1e-9)

    network = nx.Graph()
    network.add_node(0, weight=2)
    network.add_node(1, weight=2)
    network.add_node(2, weight=1)
    network.add_edge(0, 1, weight=1/2)
    network.add_edge(0, 2, weight=1/2)
    network.add_edge(1, 2, weight=1/2)

    for node in network.nodes:
        network.add_edge(node, node, weight=1e9)

    return task_graph, network

def main():
    examples: Dict[str, Callable[[], Tuple[nx.DiGraph, nx.Graph]]] = {
        'example_1': get_example_one,
        'example_2': get_example_two,
        'example_3': get_example_three,
    }

    for name, get_example in examples.items():
        task_graph, network = get_example()

        scheduler = HeftScheduler()
        schedule = scheduler.schedule(network, task_graph)

        bf_scheduler = BruteForceScheduler()
        bf_schedule = bf_scheduler.schedule(network, task_graph)

        (thisdir / name).mkdir(exist_ok=True, parents=True)

        # draw the task graph
        ax: plt.Axes = draw_task_graph(task_graph)
        ax.set_title('Task Graph')
        fig = ax.get_figure()
        fig.savefig(thisdir / name / 'task_graph.png')

        # draw the network
        ax: plt.Axes = draw_network(network)
        ax.set_title('Network')
        fig = ax.get_figure()
        fig.savefig(thisdir / name / 'network.png')

        # draw the schedule
        fig: Figure = draw_gantt(schedule)
        fig.update_layout(title='Schedule')
        fig.write_image(str(thisdir / name / 'schedule.png'))

        # draw the brute force schedule
        fig: Figure = draw_gantt(bf_schedule)
        fig.update_layout(title='Brute Force Schedule')
        fig.write_image(str(thisdir / name / 'brute_force_schedule.png'))


if __name__ == '__main__':
    main()
