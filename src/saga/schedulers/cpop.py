from functools import lru_cache
import heapq
from typing import Dict, Optional
import numpy as np

from saga import Scheduler, ScheduledTask, Schedule, Network, TaskGraph


@lru_cache(maxsize=None)
def upward_rank(network: Network, task_graph: TaskGraph) -> Dict[str, float]:
    """Computes the upward ranks of the tasks in the task graph.

    Args:
        network (Network): The network graph.
        task_graph (TaskGraph): The task graph.

    Returns:
        Dict[Hashable, float]: The upward ranks of the tasks.
    """
    ranks: Dict[str, float] = {}

    topological_order = task_graph.topological_sort()
    for task in topological_order[::-1]:
        avg_comp_time = np.mean([task.cost / node.speed for node in network.nodes])
        max_comm_time = (
            0
            if task_graph.out_degree(task.name) <= 0
            else max(
                [
                    ranks[task_graph_dependency.target]
                    + np.mean(
                        [
                            task_graph_dependency.size / network_edge.speed
                            for network_edge in network.edges
                        ]
                    )
                    for task_graph_dependency in task_graph.out_edges(task.name)
                ]
            )
        )
        ranks[task.name] = float(avg_comp_time + max_comm_time)

    return ranks


def downward_rank(network: Network, task_graph: TaskGraph) -> Dict[str, float]:
    """Computes the downward ranks of the tasks in the task graph.

    Args:
        network (Network): The network graph.
        task_graph (TaskGraph): The task graph.

    Returns:
        Dict[Hashable, float]: The downward ranks of the tasks.
    """
    ranks: Dict[str, float] = {}
    for task in task_graph.topological_sort():
        rank = (
            0
            if task_graph.in_degree(task.name) <= 0
            else max(
                [
                    ranks[task_graph_dependency.source]
                    + np.mean(
                        [
                            task_graph_dependency.size / network_edge.speed
                            for network_edge in network.edges
                        ]
                    )
                    + (
                        task.cost
                        / np.mean([neighbor.speed for neighbor in network.nodes])
                    )
                    for task_graph_dependency in task_graph.in_edges(task.name)
                ]
            )
        )
        ranks[task.name] = float(rank)
    return ranks


@lru_cache(maxsize=None)
def cpop_ranks(network: Network, task_graph: TaskGraph) -> Dict[str, float]:
    """Computes the ranks of the tasks in the task graph using for the CPoP algorithm.

    Args:
        network (Network): The network graph.
        task_graph (TaskGraph): The task graph.

    Returns:
        Dict[Hashable, float]: The ranks of the tasks.
    """
    upward_ranks = upward_rank(network, task_graph)
    downward_ranks = downward_rank(network, task_graph)
    ranks = {
        task.name: (upward_ranks[task.name] + downward_ranks[task.name])
        for task in task_graph.tasks
    }
    return ranks


class CpopScheduler(Scheduler):
    """Implements the CPoP algorithm for task scheduling.

    Source: https://dx.doi.org/10.1109/71.993206
    """

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Computes the schedule for the task graph using the CPoP algorithm.

        Args:
            network (Network): The network to schedule on.
            task_graph (TaskGraph): The task graph to schedule.
            schedule (Optional[Schedule], optional): An initial schedule to build upon. Defaults to None.
            min_start_time (float, optional): The minimum start time for tasks. Defaults to 0.0.

        Returns:
            Schedule: The resulting schedule.

        Raises:
            ValueError: If instance is invalid.
        """
        # initialise comp_schedule and task_map but if schedule is not None, use it
        comp_schedule = Schedule(task_graph, network)
        task_map: Dict[str, ScheduledTask] = {}
        if schedule is not None:
            comp_schedule = schedule.model_copy()
            task_map = {
                task.name: task for _, tasks in schedule.items() for task in tasks
            }

        ranks = cpop_ranks(network, task_graph)
        entry_tasks = [
            task.name
            for task in task_graph.tasks
            if task_graph.in_degree(task.name) == 0
        ]
        cp_rank = ranks[max(entry_tasks, key=lambda task_name: ranks[task_name])]

        # node that minimizes sum of execution times of tasks on critical path
        # this should just be the node with the highest weight
        cp_node = min(
            network.nodes,
            key=lambda node: sum(
                task.cost / node.speed
                for task in task_graph.tasks
                if np.isclose(ranks[task.name], cp_rank)
            ),
        )

        # Include tasks not scheduled yet, skip tasks that are
        pq = [
            (-ranks[task.name], task)
            for task in task_graph.tasks
            if task.name not in task_map
            and all(
                task_graph_dep.source in task_map
                for task_graph_dep in task_graph.in_edges(task.name)
            )
        ]
        heapq.heapify(pq)

        while pq:
            task_rank, task = heapq.heappop(pq)

            min_finish_time = np.inf
            best_node = cp_node  # arbitrary initialization

            nodes = network.nodes
            if np.isclose(-task_rank, cp_rank):
                nodes = frozenset([cp_node])

            for node in nodes:
                start_time = comp_schedule.get_earliest_start_time(
                    task=task, node=node, append_only=False
                )
                end_time = start_time + (task.cost / node.speed)
                if end_time < min_finish_time:
                    min_finish_time = end_time
                    best_node = node

            new_exec_time = task.cost / best_node.speed
            new_task = ScheduledTask(
                node=best_node.name,
                name=task.name,
                start=min_finish_time - new_exec_time,
                end=min_finish_time,
            )
            comp_schedule.add_task(new_task)
            task_map[task.name] = new_task

            ready_tasks = [
                task_graph.get_task(dep.target)
                for dep in task_graph.out_edges(task.name)
                if all(
                    child_dep.source in task_map
                    for child_dep in task_graph.in_edges(dep.target)
                )
            ]
            for ready_task in ready_tasks:
                heapq.heappush(pq, (-ranks[ready_task.name], ready_task))
        return comp_schedule
