from copy import deepcopy
import logging
import pathlib
from typing import Callable, Dict, Hashable, List, Optional, Tuple

import networkx as nx
import numpy as np

from ..scheduler import Scheduler, Task
from ..utils.tools import get_insert_loc

thisdir = pathlib.Path(__file__).resolve().parent


def upward_rank_scaled(network: nx.Graph,
                       task_graph: nx.DiGraph,
                       transcript_callback: Callable[[str], None] = lambda x: x) -> Dict[Hashable, float]:
    """Computes the upward rank of the tasks using scaled_weight for ranking.
    
    This version uses 'scaled_weight' attribute on task nodes for ranking calculation,
    while edges still use 'weight' for communication costs.
    
    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph with 'scaled_weight' attributes.
        transcript_callback: Optional callback for logging.
    
    Returns:
        Dict[Hashable, float]: The upward ranks of the tasks.
    """
    ranks = {}

    topological_order = list(nx.topological_sort(task_graph))
    for node in topological_order[::-1]:

        node_weight = task_graph.nodes[node]['scaled_weight']
        # rank = avg_comp_time + max(rank of successors + avg_comm_time w/ successors)
        avg_comp_time = np.mean([
            node_weight / network.nodes[neighbor]['weight']
            for neighbor in network.nodes
        ])
        transcript_callback(f"Task {node} avg comp time (scaled): {avg_comp_time:0.4f}")
        
        max_comm_time = 0 if task_graph.out_degree(node) <= 0 else max(
            [
                ranks[neighbor] + np.mean([
                    task_graph.edges[node, neighbor]['weight'] / network.edges[src, dst]['weight']
                    for src, dst in network.edges
                ])
                for neighbor in task_graph.successors(node)
            ]
        )
        transcript_callback(f"Task {node} max comm time: {max_comm_time:0.4f}")
        ranks[node] = avg_comp_time + max_comm_time
        transcript_callback(f"Task {node} rank: {ranks[node]:0.4f}")

    return ranks


def heft_rank_sort(network: nx.Graph, task_graph: nx.DiGraph) -> List[Hashable]:
    """Sort tasks based on their rank using scaled weights for ranking.

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph with 'scaled_weight' attributes.

    Returns:
        List[Hashable]: The sorted list of tasks.
    """
    rank = upward_rank_scaled(network, task_graph)
    topological_sort = {node: i for i, node in enumerate(reversed(list(nx.topological_sort(task_graph))))}
    rank = {node: (rank[node] + topological_sort[node]) for node in rank}
    #print(rank)
    return sorted(list(rank.keys()), key=rank.get, reverse=True)



class ConditionalHeftScheduler(Scheduler):
    """Schedules tasks using the HEFT algorithm.

    Source: https://dx.doi.org/10.1109/71.993206
    """

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
        if schedule is None:
            comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
            task_schedule: Dict[Hashable, Task] = {}
        else:
            comp_schedule = deepcopy(schedule)
            task_schedule = {task.name: task for node in schedule for task in schedule[node]}

        task_name: Hashable
        logging.debug("Schedule order: %s", schedule_order)
        for task_name in schedule_order:
            if task_name in task_schedule:
                continue
            min_finish_time = np.inf
            best_node = None
            for node in network.nodes:  # Find the best node to run the task
                max_arrival_time: float = max(  #
                    [
                        min_start_time,
                        *[
                            task_schedule[parent].end
                            + (
                                commtimes[(task_schedule[parent].node, node)][
                                    (parent, task_name)
                                ]
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
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_node = node, idx

            new_runtime = runtimes[best_node[0]][task_name]
            task = Task(
                best_node[0], task_name, min_finish_time - new_runtime, min_finish_time
            )
            comp_schedule[best_node[0]].insert(best_node[1], task)
            task_schedule[task_name] = task

        return comp_schedule

    def schedule(self, 
                 network: nx.Graph, 
                 task_graph: nx.DiGraph, 
                 schedule: Optional[Dict[Hashable, List[Task]]] = None,
                 min_start_time: float = 0.0) -> Dict[Hashable, List[Task]]:
        """Schedule the tasks on the network using scaled weights for ranking.
        
        This method uses 'scaled_weight' for task ranking but 'weight' for actual scheduling.
        The task_graph should have 'scaled_weight' attributes added via the add_scaled_weights
        function from bfs.py before calling this method.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph with 'scaled_weight' attributes.
            schedule (Optional[Dict[Hashable, List[Task]]], optional): The schedule. Defaults to None.
            min_start_time (float, optional): The minimum start time. Defaults to 0.0.

        Returns:
            Dict[str, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """

        # Use normal weights for scheduling (not scaled weights)
        runtimes, commtimes = ConditionalHeftScheduler.get_runtimes(network, task_graph)
        # Use scaled weights for ranking
        schedule_order = heft_rank_sort(network, task_graph)
        return self._schedule(network, task_graph, runtimes, commtimes, schedule_order, schedule, min_start_time)
