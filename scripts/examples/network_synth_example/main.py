import pathlib

import matplotlib.pyplot as plt
import networkx as nx

from saga.schedulers.heft import HeftScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

thisdir = pathlib.Path(__file__).parent.absolute()


def main():
    # simple diamond-task graph A -> B, A -> C, B -> D, C -> D
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from(['A', 'B', 'C', 'D'])
    task_graph.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
    for node in task_graph.nodes:
        task_graph.nodes[node]['weight'] = 1
    for edge in task_graph.edges:
        task_graph.edges[edge]['weight'] = 10

    task_graph.edges['A', 'B']['weight'] = 1

    # simple network with two nodes and one edge
    network = nx.Graph()
    network.add_nodes_from([1, 2], weight=1)
    network.add_edge(1, 2, weight=1/10)
    network.add_edge(1, 1, weight=1e9)
    network.add_edge(2, 2, weight=1e9)


    scheduler = HeftScheduler()
    schedule = scheduler.schedule(network, task_graph)

    ax_task_graph = draw_task_graph(task_graph, use_latex=True, figsize=(6, 8))
    # ax_task_graph.set_title('Task graph')
    ax_task_graph.get_figure().savefig('task_graph.pdf', bbox_inches='tight')
    ax_task_graph.get_figure().savefig('task_graph.png', bbox_inches='tight')
    plt.close(ax_task_graph.get_figure())

    ax_network = draw_network(network, use_latex=True, draw_colors=False)
    # ax_network.set_title('``Worse" Network')
    ax_network.get_figure().savefig('network_1.pdf', bbox_inches='tight')
    ax_network.get_figure().savefig('network_1.png', bbox_inches='tight')
    plt.close(ax_network.get_figure())

    ax_gantt = draw_gantt(schedule, use_latex=True, xmax=8)
    # ax_gantt.set_title('HEFT Schedule for ``Worse" Network')
    ax_gantt.get_figure().savefig('gantt_1.pdf', bbox_inches='tight')
    ax_gantt.get_figure().savefig('gantt_1.png', bbox_inches='tight')
    plt.close(ax_gantt.get_figure())


    # set 1 -> 2 to be faster (speed 1)
    network.edges[1, 2]['weight'] = 2

    schedule = scheduler.schedule(network, task_graph)

    ax_network = draw_network(network, use_latex=True, draw_colors=False)
    # ax_network.set_title('``Better" Network')
    ax_network.get_figure().savefig('network_2.pdf', bbox_inches='tight')
    ax_network.get_figure().savefig('network_2.png', bbox_inches='tight')
    plt.close(ax_network.get_figure())

    ax_gantt = draw_gantt(schedule, use_latex=True)
    ax_gantt.set_title('HEFT Schedule for ``Better" Network')
    ax_gantt.get_figure().savefig('gantt_2.pdf', bbox_inches='tight')
    ax_gantt.get_figure().savefig('gantt_2.png', bbox_inches='tight')
    plt.close(ax_gantt.get_figure())        



if __name__ == '__main__':
    main()