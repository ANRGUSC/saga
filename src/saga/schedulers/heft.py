from copy import deepcopy
import logging
import pathlib
from typing import Callable, Dict, Hashable, List, Optional, Tuple, Set

import networkx as nx
import numpy as np

from ..scheduler import Scheduler, Task
from ..utils.tools import get_insert_loc
from .cpop import upward_rank

thisdir = pathlib.Path(__file__).resolve().parent


def heft_rank_sort(network: nx.Graph,
                   task_graph: nx.DiGraph,
                   transcript_callback: Callable[[str], None] = lambda x: x) -> List[Hashable]:
    """Sort tasks based on their rank (as defined in the HEFT paper).

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.
        transcript_callback (Optional[Callable[[str], None]]): A callback function to call with explanations
            of the scheduling process.

    Returns:
        List[Hashable]: The sorted list of tasks.
    """
    transcript_callback("Computing upward rank of tasks")
    rank = upward_rank(network, task_graph, transcript_callback)
    topological_sort = {node: i for i, node in enumerate(reversed(list(nx.topological_sort(task_graph))))}
    rank = {node: (rank[node], topological_sort[node]) for node in rank}
    task_order = sorted(list(rank.keys()), key=rank.get, reverse=True)
    transcript_callback(f"Scheduling order: {task_order}")
    return task_order


class HeftScheduler(Scheduler):
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
        transcript_callback: Callable[[str], None] = lambda x: x,
        clusters: Optional[List[Set[Hashable]]] = None,
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
            transcript_callback (Optional[Callable[[str], None]]): A callback function to call with explanations
                of the scheduling process.
            clusters (Optional[List[Set[Hashable]]]): A list of clusters of tasks that should be scheduled to
                run on the same node.

        Returns:
            Dict[Hashable, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """
        comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}
        cluster_decisions: Dict[Hashable, Hashable] = {}
        def get_cluster(task_name: Hashable) -> Set[Hashable]:
            if clusters is None:
                return {task_name}
            for cluster in clusters:
                if task_name in cluster:
                    return cluster
            return {task_name}

        task_name: Hashable
        for task_name in schedule_order:
            transcript_callback(f"Scheduling task {task_name}")
            nodes = network.nodes
            if task_name in cluster_decisions:
                nodes = [cluster_decisions[task_name]]
            min_finish_time = np.inf
            best_node = None
            for node in nodes:  # Find the best node to run the task
                transcript_callback(f"Testing task {task_name} on node {node}")
                max_arrival_time: float = max(  [
                        task_schedule[parent].end
                        + (
                            commtimes[(task_schedule[parent].node, node)][
                                (parent, task_name)
                            ]
                        )
                        for parent in task_graph.predecessors(task_name)
                    ]
                )
                transcript_callback(f"All required predecessor data for task {task_name} would be available on node {node} at {max_arrival_time:0.4f}")

                runtime = runtimes[node][task_name]
                idx, start_time = get_insert_loc(
                    comp_schedule[node], max_arrival_time, runtime
                )
                transcript_callback(f"Earliest large enough slot for task {task_name} on node {node} is {idx} at {start_time:0.4f}")

                finish_time = start_time + runtime
                transcript_callback(f"Task {task_name} on node {node} would finish at {finish_time:0.4f}")
                if finish_time < min_finish_time:
                    if best_node is not None:
                        transcript_callback(f"This is better than the previous best finish time of {min_finish_time:0.4f} (on node {best_node[0]})")
                    min_finish_time = finish_time
                    best_node = node, idx
                else:
                    if best_node is not None:
                        transcript_callback(f"This is worse than the previous best finish time of {min_finish_time:0.4f} (on node {best_node[0]})")

            transcript_callback(f"Best node for task {task_name} is {best_node[0]}")
            new_runtime = runtimes[best_node[0]][task_name]
            task = Task(
                best_node[0], task_name, min_finish_time - new_runtime, min_finish_time
            )
            transcript_callback(f"Inserting {task}")
            comp_schedule[best_node[0]].insert(best_node[1], task)
            task_schedule[task_name] = task

            if clusters is not None:
                cluster = get_cluster(task_name)
                for task in cluster:
                    cluster_decisions[task] = best_node[0]

            transcript_callback(f"Current schedule: {comp_schedule}")

        return comp_schedule

    def schedule(
        self, network: nx.Graph, task_graph: nx.DiGraph,
        transcript_callback: Callable[[str], None] = lambda x: x,
        clusters: Optional[List[Set[Hashable]]] = None
    ) -> Dict[str, List[Task]]:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            clusters (Optional[List[Set[Hashable]]]): A list of clusters of tasks that should be scheduled to
                run on the same node.

        Returns:
            Dict[str, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """

        runtimes, commtimes = HeftScheduler.get_runtimes(network, task_graph)
        schedule_order = heft_rank_sort(network, task_graph, transcript_callback)
        return self._schedule(network, task_graph, runtimes, commtimes, schedule_order, transcript_callback, clusters)
