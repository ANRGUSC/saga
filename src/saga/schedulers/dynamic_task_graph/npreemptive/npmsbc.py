import logging
from typing import Dict, Hashable, List, Tuple, Optional

import networkx as nx
import numpy as np

from ....scheduler import Scheduler, Task
from ....utils.tools import get_insert_loc


def calulate_sbct(
    network: nx.Graph,
    task_graph: nx.DiGraph,
    runtimes: Dict[Hashable, Dict[Hashable, float]],
    commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
) -> (Dict[Hashable, float], Dict[Hashable, Hashable]):
    """
    Computes the strict bound completion time of the tasks in the task graph.

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.
        runtimes (Dict[Hashable, Dict[Hashable, float]]): A dictionary mapping nodes to a
            dictionary of tasks and their runtimes.
        commtimes (Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]): A
            dictionary mapping edges to a dictionary of task dependencies and their communication times.

    Returns:
        Dict[Hashable, float]: the strict bound completion time for each task
        Dict[Hashable, float]: the favourite node achieving the sbl time for each task
    """
    sbct = {}
    ifav = {}

    def get_drt(task_name, node):
        """
        Calculate the data ready time
        """
        return max(
            (sbct[pred] + commtimes[ifav[pred], node][pred, task_name])
            for pred in task_graph.predecessors(task_name)
        )

    def get_sbct(task_name):
        """
        Calculate the strict bound completion time for a specific task
        """
        min_val = float("inf")
        min_node = None
        degree = task_graph.in_degree(task_name)
        for node in network.nodes:
            if degree <= 0:
                temp_val = runtimes[node][task_name]
            else:
                temp_val = get_drt(task_name, node) + runtimes[node][task_name]

            if temp_val < min_val:
                min_val = temp_val
                min_node = node
        return min_val, min_node

    for task_name in nx.topological_sort(task_graph):
        sbct[task_name], ifav[task_name] = get_sbct(task_name)

    return sbct, ifav


def get_sbl(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """
    Computes the static b-level of the tasks in the task graph.

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        Dict[Hashable, float]: the static b-level for each task
    """
    sbl = {}

    is_comp_zero = all(
        np.isclose(network.nodes[_node]["weight"], 0) for _node in network.nodes
    )

    def avg_comp_time(task: Hashable) -> float:
        """Get the average compute time for a task"""
        if is_comp_zero:
            return 1e-9
        return np.mean(
            [
                task_graph.nodes[task]["weight"] / network.nodes[node]["weight"]
                for node in network.nodes
                if not np.isclose(network.nodes[node]["weight"], 0)
            ]
        )

    for task_name in nx.topological_sort(task_graph):
        sbl[task_name] = (
            0
            if task_graph.in_degree(task_name) <= 0
            else max(
                (avg_comp_time(pred) + sbl[pred])
                for pred in task_graph.predecessors(task_name)
            )
        )

    return sbl


def calculate_st(
    task_graph: nx.DiGraph,
    ifav: Dict[Hashable, float],
    runtimes: Dict[Hashable, Dict[Hashable, float]],
    commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
) -> Dict[Hashable, float]:
    """
    Calculate the start-times on ifav nodes

    Args:
        task_graph (nx.DiGraph): The task graph.
        ifav (Dict[Hashable, float]): Favourite node for each task to run on.
        runtimes (Dict[Hashable, Dict[Hashable, float]]): A dictionary mapping nodes to a
            dictionary of tasks and their runtimes.
        commtimes (Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]): A
            dictionary mapping edges to a dictionary of task dependencies and their communication times.

    Returns:
        Dict[Hashable, float]: start-timed for each tasks
    """
    st = {}
    for task_name in nx.topological_sort(task_graph):
        st[task_name] = (
            0
            if task_graph.in_degree(task_name) <= 0
            else max(
                (
                    st[pred]
                    + runtimes[ifav[pred]][pred]
                    + commtimes[ifav[pred], ifav[task_name]][pred, task_name]
                )
                for pred in task_graph.predecessors(task_name)
            )
        )
    return st


def get_priority(
    task_graph: nx.Graph,
    sbct: Dict[Hashable, float],
    sbl: Dict[Hashable, float],
    st: Dict[Hashable, float],
) -> Dict[Hashable, float]:
    """
    Calculate the priority values

    Args:
        task_graph (nx.DiGraph): The task graph.
        sbct (Dict[Hashable, float]): strict bound completion time.
        sbl (Dict[Hashable, float]): static b-level.
        st (Dict[Hashable, float]): start-time on ifav.

    Returns:
        Dict[Hashable, float]: Priority values
    """
    return {
        task_name: sbct[task_name] + sbl[task_name] + st[task_name]
        for task_name in task_graph.nodes
    }


class ResidualMsbcScheduler(Scheduler):  # pylint: disable=too-few-public-methods
    """Implements the Multiple Strict Bounds Constraints (MSBC) scheduling algorithm

    Source: https://dx.doi.org/10.1109/PACRIM.2005.1517309
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
        priorities: Dict[Hashable, float],
        current_schedule: Optional[Dict[str, List[Task]]] = None,
        task_graph_arrival_time: float = 0.0,
    ) -> Dict[Hashable, List[Task]]:
        """Computes the schedule for the task graph using the MSBC algorithm.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            runtimes (Dict[Hashable, Dict[Hashable, float]]): A dictionary mapping nodes to a
                dictionary of tasks and their runtimes.
            commtimes (Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]): A
                dictionary mapping edges to a dictionary of task dependencies and their communication times.
            priorities (Dict[Hashable, float]): Priority values for tasks

        Returns:
            Dict[Hashable, List[Task]]: The schedule for the task graph.
        """
        comp_schedule: Dict[Hashable, List[Task]] = current_schedule or {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}
        ready_set = set(
            [
                task_name
                for task_name in task_graph.nodes
                if task_graph.in_degree(task_name) <= 0
            ]
        )

        scheduled_set = set()
        sorted_nodes = sorted(
            network.nodes, key=lambda node: network.nodes[node]["weight"], reverse=True
        )
        while len(ready_set) > 0:
            task_name = max(ready_set, key=lambda x: priorities[x])
            ready_set.remove(task_name)
            min_start_time = np.inf
            best_node = None
            for node in sorted_nodes:  # Find the best node to run the task
                max_arrival_time: float = max(  #
                    [
                        task_graph_arrival_time,
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
                if start_time < min_start_time:
                    min_start_time = start_time
                    best_node = node, idx

            new_runtime = runtimes[best_node[0]][task_name]
            task = Task(
                best_node[0], task_name, min_start_time, min_start_time + new_runtime
            )
            comp_schedule[best_node[0]].insert(best_node[1], task)
            task_schedule[task_name] = task
            scheduled_set.add(task_name)
            for succ in task_graph.successors(task_name):
                is_ready = True
                for pred in task_graph.predecessors(succ):
                    if pred not in scheduled_set:
                        is_ready = False
                        break
                if is_ready:
                    ready_set.add(succ)
        return comp_schedule

    def schedule(
        self, network: nx.Graph, task_graphs: List[Tuple[nx.DiGraph, float]],
    ) -> Dict[Hashable, List[Task]]:
        """Computes the schedule for the task graph using the MSBC algorithm.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: The schedule for the task graph.

        Raises:
            ValueError: If instance is invalid.
        """

        comp_schedule: Dict[Hashable, List[Task]] = None
        # {node: [] for node in network.nodes}

        for task_graph_tupple in task_graphs:
            task_graph = task_graph_tupple[0]
            task_graph_arrival_time = task_graph_tupple[1]

            runtimes, commtimes = ResidualMsbcScheduler.get_runtimes(network, task_graph)
            sbl = get_sbl(network, task_graph)
            sbct, ifav = calulate_sbct(network, task_graph, runtimes, commtimes)
            st = calculate_st(task_graph, ifav, runtimes, commtimes)
            priorities = get_priority(task_graph, sbct, sbl, st)
            comp_schedule = self._schedule(network, task_graph, runtimes, commtimes, priorities, comp_schedule, task_graph_arrival_time)

        return comp_schedule
