import logging
import pathlib
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np

from ..base import Task
from ..base import Scheduler
from ..utils.tools import check_instance_simple, get_insert_loc

thisdir = pathlib.Path(__file__).resolve().parent

def heft_rank_sort(network: nx.Graph, task_graph: nx.DiGraph) -> List[Hashable]:
    """Sort tasks based on their rank (as defined in the HEFT paper).
    
    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        List[Hashable]: The sorted list of tasks.
    """
    rank = {}
    logging.debug(f"Topological sort: {list(nx.topological_sort(task_graph))}")
    for task_name in reversed(list(nx.topological_sort(task_graph))):
        avg_comp = np.mean([
            task_graph.nodes[task_name]['weight'] / 
            network.nodes[node]['weight'] for node in network.nodes
        ])
        max_comm = 0 if task_graph.out_degree(task_name) <= 0 else max(
            ( 
                rank.get(succ, 0) + 
                np.mean([
                    task_graph.edges[task_name, succ]['weight'] /
                    network.edges[src, dst]['weight'] for src, dst in network.edges
                ])
            )
            for succ in task_graph.successors(task_name)
        )
        rank[task_name] = avg_comp + max_comm

    return sorted(list(rank.keys()), key=rank.get, reverse=True)


class HeftScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_runtimes(network: nx.Graph, task_graph: nx.DiGraph) -> Tuple[Dict[Hashable, Dict[Hashable, float]], Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]]:
        """Get the expected runtimes of all tasks on all nodes.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Tuple[Dict[Hashable, Dict[Hashable, float]], Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]]: A tuple of dictionaries mapping nodes to a dictionary of tasks and their runtimes and edges to a dictionary of tasks and their communication times.
                The first dictionary maps nodes to a dictionary of tasks and their runtimes.
                The second dictionary maps edges to a dictionary of task dependencies and their communication times.
        """
        runtimes = {}
        for node in network.nodes:
            runtimes[node] = {}
            speed: float = network.nodes[node]["weight"]
            for task in task_graph.nodes:
                cost: float = task_graph.nodes[task]["weight"]
                runtimes[node][task] = cost / speed
                logging.debug(f"Task {task} on node {node} has runtime {runtimes[node][task]}")

        commtimes = {}
        for src, dst in network.edges:
            commtimes[src, dst] = {}
            commtimes[dst, src] = {}
            speed: float = network.edges[src, dst]["weight"]
            for src_task, dst_task in task_graph.edges:
                cost = task_graph.edges[src_task, dst_task]["weight"]
                commtimes[src, dst][src_task, dst_task] = cost / speed
                commtimes[dst, src][src_task, dst_task] = cost / speed
                logging.debug(f"Task {src_task} on node {src} to task {dst_task} on node {dst} has communication time {commtimes[src, dst][src_task, dst_task]}")

        return runtimes, commtimes

    def _schedule(self, 
                  network: nx.Graph, 
                  task_graph: nx.DiGraph,
                  runtimes: Dict[Hashable, Dict[Hashable, float]],
                  commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
                  schedule_order: List[Hashable]) -> Dict[Hashable, List[Task]]:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            runtimes (Dict[Hashable, Dict[Hashable, float]]): A dictionary mapping nodes to a dictionary of tasks and their runtimes.
            commtimes (Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]): A dictionary mapping edges to a dictionary of task dependencies and their communication times.
            schedule_order (List[Hashable]): The order in which to schedule the tasks.

        Returns:
            Dict[Hashable, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """
        comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}

        task_name: Hashable
        logging.debug(f"Schedule order: {schedule_order}")
        for task_name in schedule_order:
            min_finish_time = np.inf 
            best_node = None 
            for node in network.nodes: # Find the best node to run the task
                logging.debug(f"Testing task {task_name} on node {node}")
                max_arrival_time: float = max( # 
                    [
                        0.0, *[
                            task_schedule[parent].end + (
                                commtimes[(task_schedule[parent].node, node)][(parent, task_name)]
                            )
                            for parent in task_graph.predecessors(task_name)
                        ]
                    ]
                )

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
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[str, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """
        check_instance_simple(network, task_graph)
        runtimes, commtimes = HeftScheduler.get_runtimes(network, task_graph)
        schedule_order = heft_rank_sort(network, task_graph)
        return self._schedule(network, task_graph, runtimes, commtimes, schedule_order)