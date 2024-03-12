import logging
import pathlib

import matplotlib.pyplot as plt
import networkx as nx

from saga.schedulers import ETFScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

logging.basicConfig(level=logging.INFO)

thisdir = pathlib.Path(__file__).parent.absolute()


plt.rcParams.update({
    # 'font.size': 20,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

def main():
    """Main function"""

    network = nx.Graph()
    network.add_node("v_1", weight=1.0)
    network.add_node("v_2", weight=1.2)
    network.add_node("v_3", weight=1.5)
    network.add_edge("v_1", "v_2", weight=0.5)
    network.add_edge("v_1", "v_3", weight=1.0)
    network.add_edge("v_2", "v_3", weight=1.2)
    # add self loops
    network.add_edge("v_1", "v_1", weight=1e9)
    network.add_edge("v_2", "v_2", weight=1e9)
    network.add_edge("v_3", "v_3", weight=1e9)

    task_graph = nx.DiGraph()
    task_graph.add_node('t_1', weight=1.7)
    task_graph.add_node('t_2', weight=1.2)
    task_graph.add_node('t_3', weight=2.2)
    task_graph.add_node('t_4', weight=0.8)

    task_graph.add_edge('t_1', 't_2', weight=0.6)
    task_graph.add_edge('t_1', 't_3', weight=0.5)
    task_graph.add_edge('t_2', 't_4', weight=1.3)
    task_graph.add_edge('t_3', 't_4', weight=1.6)


    scheduler = ETFScheduler()
    schedule = scheduler.schedule(network, task_graph)

    print(schedule)

    ax: plt.Axes = draw_gantt(schedule, use_latex=True)
    ax.get_figure().savefig(str(thisdir / 'gantt.pdf'))

    # make logging level higher to remove debug messages
    logging.basicConfig(level=logging.INFO)
    ax: plt.Axes = draw_task_graph(task_graph, use_latex=True)
    ax.get_figure().savefig(str(thisdir / 'task_graph.pdf'))

    ax: plt.Figure = draw_network(network, draw_colors=False, use_latex=True)
    ax.get_figure().savefig(str(thisdir / 'network.pdf'))

    
if __name__ == '__main__':
    main()
