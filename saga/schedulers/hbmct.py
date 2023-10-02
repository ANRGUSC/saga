import logging  
import pathlib
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np

from ..scheduler import Scheduler, Task
from ..utils.tools import get_insert_loc
from .heft import heft_rank_sort

def hbmct_rank_sort(network: nx.Graph, task_graph: nx.DiGraph) -> List[Hashable]:    
    """Sort tasks based on their rank (as defined in the HEFT paper).

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        List[Hashable]: The sorted list of tasks.
    """

    return heft_rank_sort(network, task_graph)

def hbmct_create_groups(network: nx.Graph, task_graph: nx.DiGraph) -> List[List[Hashable]]:
    """Create Independent Groups for scheduling.

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        List[Hashable]: The sorted list of tasks.
    """
    rankings = hbmct_rank_sort(network, task_graph)
    groups = [[rankings[0]]]
    for task_name in rankings[1:]:
        is_same_group = True
        for predecessors in task_graph.predecessors(task_name):
            if predecessors in groups[-1]:
                groups.append([task_name])
                is_same_group = False
                break
        if is_same_group:
            groups[-1].append(task_name)
    print(groups)
    return groups

def calculate_est(network: nx.Graph,
                  task_graph: nx.DiGraph,
                  group:List[Hashable],
                  commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
                  comp_schedule: Dict[Hashable, List[Task]],
                  task_schedule: List[Hashable]) -> Dict[Hashable, Dict[Hashable, float]]:
    
    est_table = {task_name:{node:None for node in network.nodes} for task_name in task_graph.nodes}
    for task_name in group:
        for node in comp_schedule:
            max_arrival_time: float = max( 
                    [
                        0.0, *[
                            task_schedule[parent].end + (
                                commtimes[(task_schedule[parent].node, node)][(parent, task_name)]
                            )
                            for parent in task_graph.predecessors(task_name)
                        ]
                    ]
                )
            if comp_schedule[node]:
                est_table[task_name][node] = max(max_arrival_time, comp_schedule[node][-1].end)
            else:
                est_table[task_name][node] = max_arrival_time
    return est_table
  
def get_initial_assignments(network: nx.Graph,
                            runtimes: Dict[Hashable, Dict[Hashable, float]],
                            group:List[Hashable],
                            est_table:Dict[Hashable, Dict[Hashable, float]]
                            ) -> Dict[Hashable, List[Hashable]]:
    assignments = {node:[] for node in network.nodes}
    for task_name in group: #Assign nodes based on execution times
        assigned_node = min(runtimes[task_name], key= runtimes[task_name].get)
        assignments[assigned_node].append(task_name)
    for node in assignments: #sort based on earliest start times
        assignments[node].sort(key= lambda x, node=node: est_table[x][node])
    print("assignments", assignments)
    return assignments
    
                
            
class HbmctScheduler(Scheduler):
    """Schedules tasks using the HBMCT algorithm."""
    @staticmethod
    def get_runtimes(network: nx.Graph,
                     task_graph: nx.DiGraph) -> Tuple[Dict[Hashable, Dict[Hashable, float]],
                                                      Dict[Tuple[Hashable, Hashable],
                                                           Dict[Tuple[Hashable, Hashable], float]]]:
        """Get the expected runtimes of all tasks on all nodes.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Tuple[Dict[Hashable, Dict[Hashable, float]],
                  Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]]:
                A tuple of dictionaries mapping nodes to a dictionary of tasks and their runtimes
                and edges to a dictionary of tasks and their communication times. The first dictionary
                maps nodes to a dictionary of tasks and their runtimes. The second dictionary maps edges
                to a dictionary of task dependencies and their communication times.
        """
        runtimes = {}
        for node in network.nodes:
            runtimes[node] = {}
            speed: float = network.nodes[node]["weight"]
            for task in task_graph.nodes:
                cost: float = task_graph.nodes[task]["weight"]
                runtimes[node][task] = cost / speed
                logging.debug("Task %s on node %s has runtime %s", task, node, runtimes[node][task])

        commtimes = {}
        for src, dst in network.edges:
            commtimes[src, dst] = {}
            commtimes[dst, src] = {}
            speed: float = network.edges[src, dst]["weight"]
            for src_task, dst_task in task_graph.edges:
                cost = task_graph.edges[src_task, dst_task]["weight"]
                commtimes[src, dst][src_task, dst_task] = cost / speed
                commtimes[dst, src][src_task, dst_task] = cost / speed
                logging.debug(
                    "Task %s on node %s to task %s on node %s has communication time %s",
                    src_task, src, dst_task, dst, commtimes[src, dst][src_task, dst_task]
                )

        return runtimes, commtimes
    
    @staticmethod
    def schedule_groups(network: nx.Graph,
                  task_graph: nx.DiGraph,
                  groups:List[List[Hashable]],
                  runtimes: Dict[Hashable, Dict[Hashable, float]],
                  commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]):

        comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}
        for group in groups:
            est_table = calculate_est(network, task_graph, group, commtimes, comp_schedule, task_schedule)
            average_est = {task_name: np.mean([
                est for est in est_table[task_name]
            ]) for task_name in est_table}
            assignments = get_initial_assignments(network, runtimes, group, est_table)
            for node in assignments:
                for task_name in assignments[node]:
                    start_time = est_table[task_name][node]
                    if comp_schedule[node]:
                        start_time = max(start_time, comp_schedule[node][-1].end)
                    task = Task(node, task_name, start_time, start_time + runtimes[task_name][node])
                    comp_schedule[node].append(task)
                    task_schedule[task_name] = task


        
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[str, List[Task]]:
        # print("scheduling...")
        runtimes, commtimes = HbmctScheduler.get_runtimes(network, task_graph)
        groups = hbmct_create_groups(network, task_graph)
        self.schedule_groups(network, task_graph, groups, runtimes, commtimes)
        return None