import logging
import pathlib
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np

from ..base import Task
from ..base import Scheduler
from ..utils.tools import check_instance_simple, get_insert_loc

thisdir = pathlib.Path(__file__).resolve().parent

def range_suffrage_sort(network: nx.Graph, task_graph: nx.DiGraph) -> List[Hashable]:
    """Sort tasks based on the range suffrage model.

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        List[Hashable]: The sorted list of tasks.
    """
    task_ranges = {}
    for task_name in task_graph.nodes:
        start_time = min(task_graph.nodes[task_name]['start_times'])
        end_time = max(task_graph.nodes[task_name]['end_times'])
        task_ranges[task_name] = end_time - start_time

    return sorted(list(task_ranges.keys()), key=task_ranges.get, reverse=True)


class RangeSuffrageScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_runtimes(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, Dict[Hashable, float]]:
        runtimes = {}
        for node in network.nodes:
            runtimes[node] = {}
            speed: float = network.nodes[node]["weight"]
            for task in task_graph.nodes:
                cost: float = task_graph.nodes[task]["weight"]
                runtimes[node][task] = cost / speed
                logging.debug(f"Task {task} on node {node} has runtime {runtimes[node][task]}")
        return runtimes

    def _schedule(self, 
                  network: nx.Graph, 
                  task_graph: nx.DiGraph,
                  runtimes: Dict[Hashable, Dict[Hashable, float]],
                  schedule_order: List[Hashable]) -> Dict[Hashable, List[Task]]:
        comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}

        for task_name in schedule_order:
            min_finish_time = np.inf 
            best_node = None 
            for node in network.nodes:
                logging.debug(f"Testing task {task_name} on node {node}")
                max_arrival_time: float = max([
                    0.0, *[
                        task_schedule[parent].end
                        for parent in task_graph.predecessors(task_name)
                    ]
                ])

                runtime = runtimes[node][task_name]   
                idx, start_time = get_insert_loc(comp_schedule[node], max_arrival_time, runtime)
                
                finish_time = start_time + runtime
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_node = node, idx 
            
            new_runtime = runtimes[best_node[0]][task_name]
            task = Task(best_node[0], task_name, min_finish_time - new_runtime, min_finish_time)
            comp_schedule[best_node[0]].insert(best_node[1], task)
            task_schedule[task_name] = task

        return comp_schedule
    
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[str, List[Task]]:
        check_instance_simple(network, task_graph)
        runtimes = RangeSuffrageScheduler.get_runtimes(network, task_graph)
        schedule_order = range_suffrage_sort(network, task_graph)  # Use  range suffrage sorting here
        return self._schedule(network, task_graph, runtimes, schedule_order)
