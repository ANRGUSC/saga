import logging
import pathlib
from typing import Dict, List

import networkx as nx
from matplotlib import pyplot as plt

from saga.scheduler import Task
from saga.schedulers import CpopScheduler, HeftScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

logging.basicConfig(level=logging.DEBUG)
thisdir = pathlib.Path(__file__).parent.absolute()

def get_makespan(schedule: Dict[str, List[Task]]) -> float:
    """Get makespan of a schedule.
    
    Args:
        schedule (Dict[str, List[Task]]): Schedule to get makespan of.

    Returns:
        float: Makespan of schedule.
    """
    return max([0 if not tasks else tasks[-1].end for tasks in schedule.values()])

def main():
    """Main function."""
    savepath = thisdir / 'figures'
    savepath.mkdir(exist_ok=True)

    # simple diamon task graph
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from([1, 2, 3, 4, 5], weight=3)
    task_graph.add_edges_from([(1, 2), (1, 3), (1, 4)], weight=2)
    task_graph.add_edges_from([(2, 5), (3, 5), (4, 5)], weight=3)


    axis = draw_task_graph(task_graph, use_latex=False)
    axis.get_figure().savefig(savepath / 'task_graph.pdf')
    plt.close(axis.get_figure())

    # simple 3-node network (complete graph)
    network = nx.Graph()
    network.add_nodes_from([1, 2, 3], weight=1)
    network.add_edges_from([(1, 2), (1, 3), (2, 3)], weight=1)
    network.add_edges_from([(1, 1), (2, 2), (3, 3)], weight=1e9)

    network.nodes[3]['weight'] = 1 + 1e-9

    axis = draw_network(network, draw_colors=False, use_latex=False)
    axis.get_figure().savefig(savepath / 'network.pdf')
    plt.close(axis.get_figure())

    schedule_heft = HeftScheduler().schedule(network, task_graph)
    heft_makespan = get_makespan(schedule_heft)

    schedule_cpop = CpopScheduler().schedule(network, task_graph)
    cpop_makespan = get_makespan(schedule_cpop)

    # modify network
    network.edges[(1, 3)]['weight'] = 1/2
    network.edges[(2, 3)]['weight'] = 1/2

    axis = draw_network(network, draw_colors=False, use_latex=False)
    axis.get_figure().savefig(savepath / 'modified_network.pdf')
    plt.close(axis.get_figure())

    schedule_heft_modified = HeftScheduler().schedule(network, task_graph)
    heft_makespan_modified_network = get_makespan(schedule_heft_modified)

    schedule_cpop_modified = CpopScheduler().schedule(network, task_graph)
    cpop_makespan_modified_network = get_makespan(schedule_cpop_modified)

    print(f'HEFT makespan: {heft_makespan:.2f}')
    print(f'CPOP makespan: {cpop_makespan:.2f}')
    print(f'HEFT makespan (modified network): {heft_makespan_modified_network:.2f}')
    print(f'CPOP makespan (modified network): {cpop_makespan_modified_network:.2f}')

    # Draw schedules
    max_makespan = max(heft_makespan, cpop_makespan, heft_makespan_modified_network, cpop_makespan_modified_network)
    ## HEFT
    axis = draw_gantt(schedule_heft, use_latex=False, xmax=max_makespan)
    axis.get_figure().savefig(savepath / 'heft_schedule.pdf')

    ## CPOP
    axis = draw_gantt(schedule_cpop, use_latex=False, xmax=max_makespan)
    axis.get_figure().savefig(savepath / 'cpop_schedule.pdf')
    
    ## HEFT (modified network)
    axis = draw_gantt(schedule_heft_modified, use_latex=False, xmax=max_makespan)
    axis.get_figure().savefig(savepath / 'heft_schedule_modified_network.pdf')

    ## CPOP (modified network)
    axis = draw_gantt(schedule_cpop_modified, use_latex=False, xmax=max_makespan)
    axis.get_figure().savefig(savepath / 'cpop_schedule_modified_network.pdf')


if __name__ == '__main__':
    main()