import logging
from typing import Dict, Optional, Tuple

import numpy as np

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph


def calulate_sbct(
    network: Network,
    task_graph: TaskGraph,
    runtimes: Dict[str, Dict[str, float]],
    commtimes: Dict[Tuple[str, str], Dict[Tuple[str, str], float]],
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Computes the strict bound completion time of the tasks in the task graph.

    Args:
        network (Network): The network.
        task_graph (TaskGraph): The task graph.
        runtimes (Dict[str, Dict[str, float]]): A dictionary mapping nodes to a
            dictionary of tasks and their runtimes.
        commtimes (Dict[Tuple[str, str], Dict[Tuple[str, str], float]]): A
            dictionary mapping edges to a dictionary of task dependencies and their communication times.

    Returns:
        Dict[str, float]: the strict bound completion time for each task
        Dict[str, str]: the favourite node achieving the sbl time for each task
    """
    sbct: Dict[str, float] = {}
    ifav: Dict[str, str] = {}
    node_names = [node.name for node in network.nodes]

    def get_drt(task_name: str, node_name: str) -> float:
        """Calculate the data ready time"""
        in_edges = task_graph.in_edges(task_name)
        return max(
            sbct[in_edge.source]
            + commtimes[ifav[in_edge.source], node_name][in_edge.source, task_name]
            for in_edge in in_edges
        )

    def get_sbct(task_name: str) -> Tuple[float, str]:
        """Calculate the strict bound completion time for a specific task"""
        min_val = float("inf")
        min_node = node_names[0]  # arbitrary initialization
        in_edges = task_graph.in_edges(task_name)
        for node_name in node_names:
            if not in_edges:
                temp_val = runtimes[node_name][task_name]
            else:
                temp_val = (
                    get_drt(task_name, node_name) + runtimes[node_name][task_name]
                )

            if temp_val < min_val:
                min_val = temp_val
                min_node = node_name
        return min_val, min_node

    for task in task_graph.topological_sort():
        sbct[task.name], ifav[task.name] = get_sbct(task.name)

    return sbct, ifav


def get_sbl(network: Network, task_graph: TaskGraph) -> Dict[str, float]:
    """
    Computes the static b-level of the tasks in the task graph.

    Args:
        network (Network): The network.
        task_graph (TaskGraph): The task graph.

    Returns:
        Dict[str, float]: the static b-level for each task
    """
    sbl: Dict[str, float] = {}

    is_comp_zero = all(np.isclose(node.speed, 0) for node in network.nodes)

    def avg_comp_time(task_name: str) -> float:
        """Get the average compute time for a task"""
        if is_comp_zero:
            return 1e-9
        task = task_graph.get_task(task_name)
        return float(
            np.mean(
                [
                    task.cost / node.speed
                    for node in network.nodes
                    if not np.isclose(node.speed, 0)
                ]
            )
        )

    for task in task_graph.topological_sort():
        in_edges = task_graph.in_edges(task.name)
        if not in_edges:
            sbl[task.name] = 0
        else:
            sbl[task.name] = max(
                avg_comp_time(in_edge.source) + sbl[in_edge.source]
                for in_edge in in_edges
            )

    return sbl


def calculate_st(
    task_graph: TaskGraph,
    ifav: Dict[str, str],
    runtimes: Dict[str, Dict[str, float]],
    commtimes: Dict[Tuple[str, str], Dict[Tuple[str, str], float]],
) -> Dict[str, float]:
    """
    Calculate the start-times on ifav nodes

    Args:
        task_graph (TaskGraph): The task graph.
        ifav (Dict[str, str]): Favourite node for each task to run on.
        runtimes (Dict[str, Dict[str, float]]): A dictionary mapping nodes to a
            dictionary of tasks and their runtimes.
        commtimes (Dict[Tuple[str, str], Dict[Tuple[str, str], float]]): A
            dictionary mapping edges to a dictionary of task dependencies and their communication times.

    Returns:
        Dict[str, float]: start-times for each tasks
    """
    st: Dict[str, float] = {}
    for task in task_graph.topological_sort():
        in_edges = task_graph.in_edges(task.name)
        if not in_edges:
            st[task.name] = 0
        else:
            st[task.name] = max(
                st[in_edge.source]
                + runtimes[ifav[in_edge.source]][in_edge.source]
                + commtimes[ifav[in_edge.source], ifav[task.name]][
                    in_edge.source, task.name
                ]
                for in_edge in in_edges
            )
    return st


def get_priority(
    task_graph: TaskGraph,
    sbct: Dict[str, float],
    sbl: Dict[str, float],
    st: Dict[str, float],
) -> Dict[str, float]:
    """
    Calculate the priority values

    Args:
        task_graph (TaskGraph): The task graph.
        sbct (Dict[str, float]): strict bound completion time.
        sbl (Dict[str, float]): static b-level.
        st (Dict[str, float]): start-time on ifav.

    Returns:
        Dict[str, float]: Priority values
    """
    return {
        task.name: sbct[task.name] + sbl[task.name] + st[task.name]
        for task in task_graph.tasks
    }


class MsbcScheduler(Scheduler):  # pylint: disable=too-few-public-methods
    """Implements the Multiple Strict Bounds Constraints (MSBC) scheduling algorithm

    Source: https://dx.doi.org/10.1109/PACRIM.2005.1517309
    """

    @staticmethod
    def get_runtimes(
        network: Network, task_graph: TaskGraph
    ) -> Tuple[
        Dict[str, Dict[str, float]],
        Dict[Tuple[str, str], Dict[Tuple[str, str], float]],
    ]:
        """Get the expected runtimes of all tasks on all nodes."""
        runtimes: Dict[str, Dict[str, float]] = {}
        for node in network.nodes:
            runtimes[node.name] = {}
            for task in task_graph.tasks:
                runtimes[node.name][task.name] = task.cost / node.speed
                logging.debug(
                    "Task %s on node %s has runtime %s",
                    task.name,
                    node.name,
                    runtimes[node.name][task.name],
                )

        commtimes: Dict[Tuple[str, str], Dict[Tuple[str, str], float]] = {}
        for edge in network.edges:
            src, dst = edge.source, edge.target
            if (src, dst) not in commtimes:
                commtimes[src, dst] = {}
            if (dst, src) not in commtimes:
                commtimes[dst, src] = {}
            for dep in task_graph.dependencies:
                commtimes[src, dst][dep.source, dep.target] = dep.size / edge.speed
                commtimes[dst, src][dep.source, dep.target] = dep.size / edge.speed
                logging.debug(
                    "Task %s on node %s to task %s on node %s has communication time %s",
                    dep.source,
                    src,
                    dep.target,
                    dst,
                    commtimes[src, dst][dep.source, dep.target],
                )

        return runtimes, commtimes

    def _schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        runtimes: Dict[str, Dict[str, float]],
        commtimes: Dict[Tuple[str, str], Dict[Tuple[str, str], float]],
        priorities: Dict[str, float],
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Computes the schedule for the task graph using the MSBC algorithm."""
        comp_schedule = Schedule(task_graph, network)
        task_schedule: Dict[str, ScheduledTask] = {}

        if schedule is not None:
            comp_schedule = schedule.model_copy()
            task_schedule = {t.name: t for _, tasks in schedule.items() for t in tasks}

        ready_set = set(
            task.name
            for task in task_graph.tasks
            if task.name not in task_schedule and task_graph.in_degree(task.name) == 0
        )

        scheduled_set = set(task_schedule.keys())
        sorted_nodes = sorted(
            [node.name for node in network.nodes],
            key=lambda node_name: network.get_node(node_name).speed,
            reverse=True,
        )

        while len(ready_set) > 0:
            task_name = max(ready_set, key=lambda x: priorities[x])
            ready_set.remove(task_name)
            best_start_time = np.inf
            best_node: str = sorted_nodes[0]

            for node_name in sorted_nodes:
                start_time = comp_schedule.get_earliest_start_time(
                    task=task_name, node=node_name, append_only=True
                )
                if start_time < best_start_time:
                    best_start_time = start_time
                    best_node = node_name

            new_runtime = runtimes[best_node][task_name]
            task = ScheduledTask(
                node=best_node,
                name=task_name,
                start=best_start_time,
                end=best_start_time + new_runtime,
            )
            comp_schedule.add_task(task)
            task_schedule[task_name] = task
            scheduled_set.add(task_name)

            for out_edge in task_graph.out_edges(task_name):
                succ = out_edge.target
                if all(
                    in_edge.source in scheduled_set
                    for in_edge in task_graph.in_edges(succ)
                ):
                    ready_set.add(succ)

        return comp_schedule

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Computes the schedule for the task graph using the MSBC algorithm.

        Args:
            network (Network): The network.
            task_graph (TaskGraph): The task graph.
            schedule (Optional[Schedule]): Optional initial schedule. Defaults to None.
            min_start_time (float): Minimum start time. Defaults to 0.0.

        Returns:
            Schedule: The schedule for the task graph.
        """
        runtimes, commtimes = MsbcScheduler.get_runtimes(network, task_graph)
        sbl = get_sbl(network, task_graph)
        sbct, ifav = calulate_sbct(network, task_graph, runtimes, commtimes)
        st = calculate_st(task_graph, ifav, runtimes, commtimes)
        priorities = get_priority(task_graph, sbct, sbl, st)
        return self._schedule(
            network,
            task_graph,
            runtimes,
            commtimes,
            priorities,
            schedule,
            min_start_time,
        )
