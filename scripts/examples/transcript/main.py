from functools import lru_cache
import logging
import pathlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from saga import ScheduledTask
from saga.schedulers import HeftScheduler, CpopScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

logging.basicConfig(level=logging.INFO)

thisdir = pathlib.Path(__file__).parent.absolute()

@lru_cache(maxsize=None)
def get_instance() -> Tuple[nx.Graph, nx.DiGraph]:
    network = nx.Graph()
    network.add_node("1", weight=1)
    network.add_node("2", weight=2)
    network.add_node("3", weight=2)
    network.add_edge("1", "2", weight=2)
    network.add_edge("1", "3", weight=2)
    network.add_edge("2", "3", weight=1)
    # add self loops
    network.add_edge("1", "1", weight=1e9)
    network.add_edge("2", "2", weight=1e9)
    network.add_edge("3", "3", weight=1e9)

    task_graph = nx.DiGraph()
    task_graph.add_node('A', weight=1)
    task_graph.add_node('B', weight=3)
    task_graph.add_node('C', weight=2)
    task_graph.add_node('D', weight=1)

    task_graph.add_edge('A', 'B', weight=1)
    task_graph.add_edge('A', 'C', weight=1)
    task_graph.add_edge('B', 'D', weight=5)
    task_graph.add_edge('C', 'D', weight=5)

    return network, task_graph

def draw_instance(network: nx.Graph, task_graph: nx.DiGraph):
    logging.basicConfig(level=logging.INFO)
    ax: plt.Axes = draw_task_graph(task_graph, use_latex=True)
    ax.get_figure().savefig(str(thisdir / 'task_graph.png'))

    ax: plt.Figure = draw_network(network, draw_colors=False, use_latex=True)
    ax.get_figure().savefig(str(thisdir / 'network.png'))

def draw_schedule(schedule: Dict[str, List[ScheduledTask]], name: str, xmax: float = None):
    ax: plt.Axes = draw_gantt(schedule, use_latex=True, xmax=xmax)
    ax.get_figure().savefig(str(thisdir / f'{name}.png'))

def heft_schedule():
    network, task_graph = get_instance()
    scheduler = HeftScheduler()
    schedule = scheduler.schedule(network, task_graph)
    return schedule

def main():
    network, task_graph = get_instance()
    scheduler_heft = HeftScheduler()
    scheduler_cpop = CpopScheduler()
    print("Scheduling with HEFT")
    schedule_heft = scheduler_heft.schedule(network, task_graph, transcript_callback=print)
    print("\n\nScheduling with CPOP")
    schedule_cpop = scheduler_cpop.schedule(network, task_graph, transcript_callback=print)


if __name__ == '__main__':
    main()
