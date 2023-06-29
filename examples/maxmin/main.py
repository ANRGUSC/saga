import itertools
from saga.common.maxmin import MaxMinScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from plotly.graph_objects import Figure 
import pathlib

thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    # create a diamond-task graph with random task weights between 0 and 1
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from([1, 2, 3, 4])
    task_graph.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

    runtimes = {}
    commtimes = {}
    machine_capacities = {}
    task_resources = {}

    for node in task_graph.nodes:
        task_graph.nodes[node]['weight'] = np.random.rand()
        runtimes[node] = {task: np.random.rand() for task in task_graph.nodes}
        task_resources[node] = np.random.rand()

    for edge in task_graph.edges:
        task_graph.edges[edge]['weight'] = np.random.rand()
        commtimes[edge] = {(task1, task2): np.random.rand() for task1 in task_graph.nodes for task2 in task_graph.nodes if task1 != task2}

    # generate a fully-connected network with random node and edge weights between 0 and 1
    network = nx.Graph()
    network.add_nodes_from([1, 2, 3, 4])

    for node in network.nodes:
        network.nodes[node]['weight'] = np.random.rand()
        machine_capacities[node] = np.random.rand()

    for (src, dst) in itertools.product(network.nodes, network.nodes):
        if src != dst:
            network.add_edge(src, dst, weight=np.random.rand())
        else:
            network.add_edge(src, dst, weight=1e9)

    scheduler = MaxMinScheduler(runtimes, commtimes, machine_capacities, task_resources)
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
    fig: Figure = draw_gantt(schedule)
    fig.update_layout(title='Schedule')
    fig.write_image(str(thisdir / 'schedule.png'))

if __name__ == '__main__':
    main()
