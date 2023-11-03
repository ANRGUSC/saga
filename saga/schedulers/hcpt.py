from typing import Dict, Hashable, List, Tuple
import logging
from collections import deque
import networkx as nx
import numpy as np
from ..scheduler import Scheduler, Task


def get_aest(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """
    Calculate the Average Earliest Start Time
    Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

    Returns:
        Dict[Hashable, float]: task-wise aest

    """
    aest = {}
    is_comm_zero = all(
        np.isclose(network.edges[_edge]["weight"], 0) for _edge in network.edges
    )
    is_comp_zero = all(
        np.isclose(network.nodes[_node]["weight"], 0) for _node in network.nodes
    )

    def avg_comm_time(parent: Hashable, child: Hashable) -> float:
        if is_comm_zero:
            return 1e-9
        return np.mean(
            [  # average communication time for output data of predecessor
                task_graph.edges[parent, child]["weight"]
                / network.edges[src, dst]["weight"]
                for src, dst in network.edges
                if not np.isclose(network.edges[src, dst]["weight"], 0)
            ]
        )

    def avg_comp_time(task: Hashable) -> float:
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
        aest[task_name] = (
            0
            if task_graph.in_degree(task_name) <= 0
            else max(
                (avg_comp_time(pred) + aest[pred] + avg_comm_time(pred, task_name))
                for pred in task_graph.predecessors(task_name)
            )
        )

    return aest


def get_alst(
    network: nx.Graph, task_graph: nx.DiGraph, aest: Dict[Hashable, float]
) -> Dict[Hashable, float]:
    """Calculate the Average Latest Start Time
    Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            aest (Dict[Hashable, float]): Average Earliest Start Times

    Returns:
        Dict[Hashable, float]: task-wise alst

    """
    alst = {}
    is_comm_zero = all(
        np.isclose(network.edges[_edge]["weight"], 0) for _edge in network.edges
    )
    is_comp_zero = all(
        np.isclose(network.nodes[_node]["weight"], 0) for _node in network.nodes
    )

    def avg_comm_time(parent: Hashable, child: Hashable) -> float:
        if is_comm_zero:
            return 1e-9
        return np.mean(
            [  # average communication time for output data of predecessor
                task_graph.edges[parent, child]["weight"]
                / network.edges[src, dst]["weight"]
                for src, dst in network.edges
                if not np.isclose(network.edges[src, dst]["weight"], 0)
            ]
        )

    def avg_comp_time(task: Hashable) -> float:
        if is_comp_zero:
            return 1e-9
        return np.mean(
            [
                task_graph.nodes[task]["weight"] / network.nodes[node]["weight"]
                for node in network.nodes
                if not np.isclose(network.nodes[node]["weight"], 0)
            ]
        )

    for task_name in reversed(list(nx.topological_sort(task_graph))):
        alst[task_name] = (
            aest[task_name]
            if task_graph.out_degree(task_name) <= 0
            else min(
                (
                    alst[succ]
                    - avg_comm_time(  # rank of successor
                        task_name, succ
                    )  # average communication time for output data of task
                )
                - avg_comp_time(task_name)
                for succ in task_graph.successors(task_name)
            )
        )
        # alst[task_name] = avg_comp_time(task_name) + max_comm
    return alst


def get_critical_nodes(
    task_graph: nx.DiGraph,
    aest: Dict[Hashable, float],
    alst: Dict[Hashable, float],
) -> List[Hashable]:
    """
    Get the critical nodes where aest = alst

    Args:
        task_graph (nx.DiGraph): The task graph.
        aest (Dict[Hashable, float]): Average Earliest Start Times
        alst (Dict[Hashable, float]): Average Latest Start Times
    Returns:
        List[Hashable]: List of critical nodes sorted by alst
    """
    critical_nodes = []
    for task_name in task_graph.nodes:
        if np.isclose(aest[task_name], alst[task_name]):
            critical_nodes.append(task_name)
    critical_nodes.sort(key=lambda x: alst[x])
    return critical_nodes


class HcptScheduler(Scheduler):  # pylint: disable=too-few-public-methods
    """Implements the HCPT algorithm for task scheduling.
    Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1267650

    Attributes:
        name (str): The name of the scheduler.
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
        critical_nodes: List[Hashable],
    ) -> Dict[Hashable, List[Task]]:
        """
        Internal schedule function

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            groups (List[List[Hashable]]): List of groups
            runtimes (Dict[Hashable, Dict[Hashable, float]]): Runtimes of each node for given tasks
            commtimes (Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]): Communication times for node-pair task-pair
            critical_nodes (List[Hashable]): Nodes with aest = alst
        Returns:
            Dict[Hashable, List[Task]]: The schedule for each node
        """

        comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}
        critical_nodes = deque(critical_nodes)
        l_queue = deque([])
        machine_rt = {node: 0 for node in network.nodes}

        def get_eeft(task_name, node):
            start = (
                0
                if task_graph.in_degree(task_name) <= 0
                else max(
                    (
                        task_schedule[pred].end
                        + commtimes[task_schedule[pred].node, node][pred, task_name]
                        for pred in task_graph.predecessors(task_name)
                    ),
                )
            )
            start = max(start, machine_rt[node])
            return start, start + runtimes[node][task_name]

        while critical_nodes:
            parents_listed = True
            for pred in task_graph.predecessors(critical_nodes[0]):
                if pred not in l_queue:
                    critical_nodes.appendleft(pred)
                    parents_listed = False
                    break
            if parents_listed:
                l_queue.append(critical_nodes.popleft())

        logging.debug("q: %s", l_queue)

        while l_queue:
            task_name = l_queue.popleft()
            start_time = finish_time = float("inf")
            min_node = None
            for node in network.nodes:
                start_time_temp, finish_time_temp = get_eeft(task_name, node)
                if finish_time_temp < finish_time:
                    start_time, finish_time = start_time_temp, finish_time_temp
                    min_node = node
            task = Task(min_node, task_name, start_time, finish_time)
            comp_schedule[min_node].append(task)
            task_schedule[task_name] = task
            machine_rt[min_node] = finish_time
            logging.debug(
                "task %s scheduled on node %s with start time = %s and finish time = %s",
                task_name,
                min_node,
                start_time,
                finish_time,
            )
        return comp_schedule

    def schedule(
        self, network: nx.Graph, task_graph: nx.DiGraph
    ) -> Dict[Hashable, List[Task]]:
        """Computes the schedule for the task graph using the HCPT algorithm.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[str, List[Task]]: The schedule for the task graph.
        """
        runtimes, commtimes = HcptScheduler.get_runtimes(network, task_graph)
        aest = get_aest(network, task_graph)
        logging.debug("aest: %s", aest)
        alst = get_alst(network, task_graph, aest)
        logging.debug("alst: %s", alst)
        critical_nodes = get_critical_nodes(task_graph, aest, alst)
        logging.debug("critical node: %s", critical_nodes)
        comp_schedule = self._schedule(
            network, task_graph, runtimes, commtimes, critical_nodes
        )
        logging.debug("final schedule %s", comp_schedule)
        return comp_schedule
