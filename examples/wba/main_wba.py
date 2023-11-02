import itertools
from saga.schedulers.wba import WBAScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pathlib

thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    # create a diamond-task graph with random task weights between 0 and 1
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from([1, 2, 3, 4])
    task_graph.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
    for node in task_graph.nodes:
        task_graph.nodes[node]['weight'] = np.random.rand()
    for edge in task_graph.edges:
        task_graph.edges[edge]['weight'] = np.random.rand()

    # generate a fully-connected network with random node and edge weights between 0 and 1
    network = nx.Graph()
    network.add_nodes_from([1, 2, 3, 4])
    for node in network.nodes:
        network.nodes[node]['weight'] = np.random.rand()
    for (src, dst) in itertools.product(network.nodes, network.nodes):
        print(f"Adding edge {src} -> {dst}")
        if src != dst:
            network.add_edge(src, dst, weight=np.random.rand())
        else:
            network.add_edge(src, dst, weight=1e9)

    scheduler = WBAScheduler()
    schedule = scheduler.schedule(network, task_graph)

    # draw the task graph
    ax: plt.Axes = draw_task_graph(task_graph)
    ax.set_title('Task Graph')
    fig = ax.get_figure()
    fig.savefig(thisdir / 'task_graph.png')

    # draw the network
    ax: plt.Axes = draw_network(network)
    ax.set_title('Network')
    fig = ax.get_figure()
    fig.savefig(thisdir / 'network.png')

    # draw the schedule
    ax: plt.Axes = draw_gantt(schedule)
    ax.set_title('Schedule')
    fig = ax.get_figure()
    fig.savefig(thisdir / 'schedule.png')

if __name__ == '__main__':
    main()
