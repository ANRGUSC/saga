from copy import deepcopy
import logging
import pathlib
from typing import Dict, Hashable, List, Optional, Tuple

import networkx as nx
import numpy as np
from queue import PriorityQueue

from ..scheduler import Scheduler, Task
from ..utils.tools import get_insert_loc, should_duplicate
from .cpop import upward_rank

thisdir = pathlib.Path(__file__).resolve().parent


def heft_rank_sort(network: nx.Graph, task_graph: nx.DiGraph) -> List[Hashable]:
    """Sort tasks based on their rank (as defined in the HEFT paper).

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        List[Hashable]: The sorted list of tasks.
    """
    rank = upward_rank(network, task_graph)
    topological_sort = {node: i for i, node in enumerate(reversed(list(nx.topological_sort(task_graph))))}
    rank = {node: (rank[node] + topological_sort[node]) for node in rank}
    return sorted(list(rank.keys()), key=rank.get, reverse=True)


class HeftScheduler(Scheduler):
    """Schedules tasks using the HEFT algorithm.

    Source: https://dx.doi.org/10.1109/71.993206
    """

    def __init__(self, duplicate_factor: int = 1) -> None:
        super().__init__()
        self.duplicate_factor = duplicate_factor # Can a task be duplicated on multiple nodes?

    @staticmethod
    def get_runtimes(
        network: nx.Graph, task_graph: nx.DiGraph
    ) -> Tuple[
        Dict[Hashable, Dict[Hashable, float]],
        Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
    ]:
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
                logging.debug(
                    "Task %s on node %s has runtime %s",
                    task,
                    node,
                    runtimes[node][task],
                )

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
                    src_task,
                    src,
                    dst_task,
                    dst,
                    commtimes[src, dst][src_task, dst_task],
                )

        return runtimes, commtimes

    def _schedule(
        self,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        runtimes: Dict[Hashable, Dict[Hashable, float]],
        commtimes: Dict[
            Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]
        ],
        schedule_order: List[Hashable],
        schedule: Optional[Dict[Hashable, List[Task]]] = None,
        min_start_time: float = 0.0,
    ) -> Dict[Hashable, List[Task]]:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            runtimes (Dict[Hashable, Dict[Hashable, float]]): A dictionary mapping nodes to a
                dictionary of tasks and their runtimes.
            commtimes (Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]): A
                dictionary mapping edges to a dictionary of task dependencies and their communication times.
            schedule_order (List[Hashable]): The order in which to schedule the tasks.
            schedule (Optional[Dict[Hashable, List[Task]]], optional): The schedule. Defaults to None.

        Returns:
            Dict[Hashable, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """
        comp_schedule: Dict[Hashable, List[Task]]
        task_schedule: Dict[Hashable, List[Task]]
        if schedule is None:
            comp_schedule = {node: [] for node in network.nodes} # node -> List[Task]
            task_schedule = {}                                   # task_name -> List[Task] (with same name), to support duplication
        else:
            comp_schedule = deepcopy(schedule)
            task_schedule = {}
            for node in comp_schedule:
                for task in comp_schedule[node]:
                    task_schedule.setdefault(task.name, []).append(task)

        task_name: Hashable
        logging.debug("Schedule order: %s", schedule_order)
        for task_name in schedule_order:
            if task_name in task_schedule:
                continue
            # duplicate_factor = 1 if task_graph.out_degree(task_name) <= 1 else self.duplicate_factor #checks for duplicates only if task has multiple children
            if should_duplicate(task_name, task_graph, network, runtimes, commtimes):
                duplicate_factor = max(self.duplicate_factor, task_graph.out_degree(task_name))
            else:
                duplicate_factor = 1
                
            best_nodes = PriorityQueue()
            for node in network.nodes:  # Find the best node to run the task
                max_arrival_time: float = max(  #Can only start after all parents, take max 
                    [
                        min_start_time, #lower bound on when a task can start 
                        *[
                            min( # the earliest time data can arrive from any of the duplicates of the parent task
                                task.end + commtimes[(task.node, node)][(parent, task_name)]
                                for task in task_schedule[parent]
                            )
                            for parent in task_graph.predecessors(task_name)
                        ],
                    ]
                )

                runtime = runtimes[node][task_name]
                idx, start_time = get_insert_loc(
                    comp_schedule[node], max_arrival_time, runtime
                )

                logging.debug(
                    "Testing task %s on node %s: start time %s, finish time %s",
                    task_name,
                    node,
                    start_time,
                    start_time + runtime,
                )

                finish_time = start_time + runtime
                best_nodes.put((finish_time, (node, idx)))

            for _ in range(duplicate_factor):
                if best_nodes.empty():
                    break
                finish_time, (node, idx) = best_nodes.get()
                runtime = runtimes[node][task_name]
                task = Task(node, task_name, finish_time - runtime, finish_time)
                comp_schedule[node].insert(idx, task)
                task_schedule.setdefault(task_name, []).append(task)

        return comp_schedule

    def schedule(self, 
                 network: nx.Graph, 
                 task_graph: nx.DiGraph, 
                 schedule: Optional[Dict[Hashable, List[Task]]] = None,
                 min_start_time: float = 0.0) -> Dict[Hashable, List[Task]]:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (Optional[Dict[Hashable, List[Task]]], optional): The schedule. Defaults to None.
            min_start_time (float, optional): The minimum start time. Defaults to 0.0.

        Returns:
            Dict[str, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """

        runtimes, commtimes = HeftScheduler.get_runtimes(network, task_graph)
        schedule_order = heft_rank_sort(network, task_graph)
        return self._schedule(network, task_graph, runtimes, commtimes, schedule_order, schedule, min_start_time)
