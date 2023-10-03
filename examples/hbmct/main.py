import itertools
from saga.schedulers.hbmct import HbmctScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from plotly.graph_objects import Figure 
import pathlib

thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from([1, 2, 3, 4])
    task_graph.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
    for node in task_graph.nodes:
        task_graph.nodes[node]['weight'] = np.random.uniform(low = 2, high= 5)
    for edge in task_graph.edges:
        task_graph.edges[edge]['weight'] = np.random.rand()

    # generate a fully-connected network with random node and edge weights between 0 and 1
    network = nx.Graph()
    network.add_nodes_from([1, 2, 3, 4])
    for node in network.nodes:
        network.nodes[node]['weight'] = np.random.rand()
    for (src, dst) in itertools.product(network.nodes, network.nodes):
        # print(f"Adding edge {src} -> {dst}")
        if src != dst:
            network.add_edge(src, dst, weight=np.random.rand())
        else:
            network.add_edge(src, dst, weight=1e9)

    scheduler = HbmctScheduler()
    schedule = scheduler.schedule(network, task_graph)
    # print(schedule)
    ax: plt.Axes = draw_task_graph(task_graph)
    ax.set_title('Task Graph')
    fig = ax.get_figure()
    fig.savefig(thisdir / 'task_graph_rand.png')

    # draw the network
    ax: plt.Axes = draw_network(network)
    ax.set_title('Network')
    fig = ax.get_figure()
    fig.savefig(thisdir / 'network.png')

if __name__ == '__main__':
    main()