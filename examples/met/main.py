import itertools
from saga.schedulers.met import METScheduler
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

    # Assign random weights to nodes and edges in the task graph
    for node in task_graph.nodes:
        task_graph.nodes[node]['weight'] = np.random.rand()

    task_graph.add_edges_from([(1, 2, {'weight': np.random.rand()}), 
                               (1, 3, {'weight': np.random.rand()}), 
                               (2, 4, {'weight': np.random.rand()}), 
                               (3, 4, {'weight': np.random.rand()})])

    # generate a fully-connected network with random node weights between 0 and 1
    network = nx.Graph()
    network.add_nodes_from([1, 2, 3, 4])

    # Assign random weights to nodes in the network
    for node in network.nodes:
        network.nodes[node]['weight'] = np.random.rand()

    for (src, dst) in itertools.product(network.nodes, network.nodes):
        if src != dst:
            network.add_edge(src, dst, weight=np.random.rand())
        else:
            network.add_edge(src, dst, weight=1e9)

    # Define machine capacities
    machine_capacities = {node: np.random.rand() for node in network.nodes}
    
    scheduler = METScheduler(machine_capacities)
    schedule = scheduler.schedule(network, task_graph)

    # draw the task graph
    ax: plt.Axes = draw_task_graph(task_graph)
    ax.set_title('Task Graph')
    fig = ax.get_figure()
    fig.savefig(thisdir / 'task_graph1.png')

    # draw the network
    ax: plt.Axes = draw_network(network)
    ax.set_title('Network')
    fig = ax.get_figure()
    fig.savefig(thisdir / 'network1.png')

    # draw the schedule
    fig: Figure = draw_gantt(schedule)
    fig.update_layout(title='Schedule')
    fig.write_image(str(thisdir / 'schedule1.png'))

if __name__ == '__main__':
    main()
