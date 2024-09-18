from saga.schedulers import ETFScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
import networkx as nx
import pathlib
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    """Main function"""

    network = nx.Graph()
    comm_speed = 1/5
    network.add_node(1, weight=1.0)
    network.add_node(2, weight=1.0)
    network.add_edge(1, 2, weight=comm_speed)
    # add self loops
    network.add_edge(1, 1, weight=1e9)
    network.add_edge(2, 2, weight=1e9)

    task_graph = nx.DiGraph()
    task_graph.add_node('A', weight=1.0)
    task_graph.add_node('B', weight=1.0)
    task_graph.add_node('C', weight=1.0)
    task_graph.add_node('D', weight=1.0)

    task_graph.add_edge('A', 'B', weight=1e-9)
    task_graph.add_edge('A', 'C', weight=1e-9)
    task_graph.add_edge('B', 'D', weight=1.0)
    task_graph.add_edge('C', 'D', weight=1.0)


    scheduler = ETFScheduler()
    schedule = scheduler.schedule(network, task_graph)

    print(schedule)

    ax: plt.Axes = draw_gantt(schedule, use_latex=True)
    ax.get_figure().savefig(str(thisdir / 'gantt.pdf'))
    plt.close(ax.get_figure())

    network.edges[1, 2]['weight'] = "$x$"

    # make logging level higher to remove debug messages
    logging.basicConfig(level=logging.INFO)
    ax: plt.Axes = draw_task_graph(task_graph, use_latex=True)
    ax.get_figure().savefig(str(thisdir / 'task_graph.pdf'))
    plt.close(ax.get_figure())

    ax: plt.Figure = draw_network(network, draw_colors=False, use_latex=True)
    ax.get_figure().savefig(str(thisdir / 'network.pdf'))
    plt.close(ax.get_figure())

    
if __name__ == '__main__':
    main()
