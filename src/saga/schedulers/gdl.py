from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph


class GDLScheduler(Scheduler):
    """Generalized Dynamic Level Scheduler

    Source: https://doi.org/10.1109/71.207593
    Notes: Considers homogenous communication speeds (not homogenous compute speeds, though)
    """

    def __init__(self, dynamic_level: int = 2):
        super().__init__()
        self.dynamic_level = dynamic_level
        if dynamic_level not in (1, 2):
            raise ValueError("dynamic_level must be 1 or 2")

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        comp_schedule = Schedule(task_graph, network)
        scheduled_tasks: Dict[str, ScheduledTask] = {}

        if schedule is not None:
            comp_schedule = schedule.model_copy()
            scheduled_tasks = {
                t.name: t for _, tasks in schedule.items() for t in tasks
            }

        # Precompute execution times
        execution_time: Dict[Tuple[str, str], float] = {}
        for task in task_graph.tasks:
            for node in network.nodes:
                execution_time[task.name, node.name] = task.cost / node.speed

        # Precompute communication times
        communication_time: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float] = {}
        for dep in task_graph.dependencies:
            for edge in network.edges:
                link = (
                    (edge.source, edge.target)
                    if edge.source < edge.target
                    else (edge.target, edge.source)
                )
                communication_time[(dep.source, dep.target), link] = (
                    dep.size / edge.speed
                )

        median_execution_time_per_task = {
            task.name: float(
                np.median(
                    [execution_time[task.name, node.name] for node in network.nodes]
                )
            )
            for task in task_graph.tasks
        }

        median_execution_time = float(
            np.median(
                [
                    execution_time[task.name, node.name]
                    for task in task_graph.tasks
                    for node in network.nodes
                ]
            )
        )

        delta_execution_time: Dict[Tuple[str, str], float] = {}
        for task in task_graph.tasks:
            for node in network.nodes:
                delta_execution_time[task.name, node.name] = (
                    median_execution_time - execution_time[task.name, node.name]
                )

        # Static level computation
        static_level: Dict[str, float] = {}
        for task in reversed(task_graph.topological_sort()):
            out_edges = task_graph.out_edges(task.name)
            if not out_edges:
                static_level[task.name] = median_execution_time_per_task[task.name]
            else:
                static_level[task.name] = median_execution_time_per_task[
                    task.name
                ] + max(
                    static_level[out_edge.target] + out_edge.size
                    for out_edge in out_edges
                )

        node_names = [node.name for node in network.nodes]

        @lru_cache(maxsize=None)
        def data_available(task_name: str, node_name: str) -> float:
            """returns time when data is available to execute task on node"""
            in_edges = task_graph.in_edges(task_name)
            if not in_edges:
                return min_start_time
            return max(
                scheduled_tasks[in_edge.source].end
                + (
                    in_edge.size
                    / network.get_edge(
                        scheduled_tasks[in_edge.source].node, node_name
                    ).speed
                )
                for in_edge in in_edges
            )

        @lru_cache(maxsize=None)
        def node_available(node_name: str) -> float:
            """returns time when node is available to execute task"""
            tasks = comp_schedule[node_name]
            if not tasks:
                return min_start_time
            return max(task.end for task in tasks)

        @lru_cache(maxsize=None)
        def dynamic_level_1(task_name: str, node_name: str) -> float:
            return (
                static_level[task_name]
                - max(data_available(task_name, node_name), node_available(node_name))
                + delta_execution_time[task_name, node_name]
            )

        largest_output_descendants: Dict[str, str] = {}
        for task in task_graph.tasks:
            out_edges = task_graph.out_edges(task.name)
            if out_edges:
                largest_output_descendants[task.name] = max(
                    out_edges, key=lambda e: e.size
                ).target

        @lru_cache(maxsize=None)
        def descendant_earliest_finish(
            task_name: str, child: str, node_name: str
        ) -> float:
            """returns earliest finish time of child's descendants on node"""
            if len(node_names) == 1:
                return np.inf
            return min(
                communication_time[
                    (task_name, child),
                    (node_name, other) if node_name < other else (other, node_name),
                ]
                + execution_time[child, other]
                for other in node_names
                if node_name != other
            )

        @lru_cache(maxsize=None)
        def descendant_consideration(task_name: str, node_name: str) -> float:
            dc = largest_output_descendants.get(task_name)
            if dc is None:
                return 0
            return median_execution_time_per_task[dc] - min(
                execution_time[dc, node_name],
                descendant_earliest_finish(task_name, dc, node_name),
            )

        @lru_cache(maxsize=None)
        def dynamic_level_2(task_name: str, node_name: str) -> float:
            return dynamic_level_1(task_name, node_name) + descendant_consideration(
                task_name, node_name
            )

        dynamic_level_func = (
            dynamic_level_1 if self.dynamic_level == 1 else dynamic_level_2
        )

        @lru_cache(maxsize=None)
        def preferred_node(task_name: str) -> str:
            return min(
                node_names,
                key=lambda node_name: dynamic_level_func(task_name, node_name),
            )

        @lru_cache(maxsize=None)
        def cost(task_name: str) -> float:
            return dynamic_level_func(task_name, preferred_node(task_name)) - max(
                dynamic_level_func(task_name, other) for other in node_names
            )

        @lru_cache(maxsize=None)
        def global_dynamic_level(task_name: str) -> float:
            return dynamic_level_func(task_name, preferred_node(task_name)) + cost(
                task_name
            )

        def clear_caches():
            """Clears the scheduler's caches"""
            data_available.cache_clear()
            node_available.cache_clear()
            dynamic_level_1.cache_clear()
            dynamic_level_2.cache_clear()
            descendant_earliest_finish.cache_clear()
            descendant_consideration.cache_clear()
            dynamic_level_func.cache_clear()
            cost.cache_clear()
            global_dynamic_level.cache_clear()
            preferred_node.cache_clear()

        num_tasks = len(list(task_graph.tasks))
        while len(scheduled_tasks) < num_tasks:
            candidate_tasks = [
                task.name
                for task in task_graph.tasks
                if task.name not in scheduled_tasks
                and all(
                    in_edge.source in scheduled_tasks
                    for in_edge in task_graph.in_edges(task.name)
                )
            ]
            task_name = min(candidate_tasks, key=global_dynamic_level)
            node_name = preferred_node(task_name)
            start_time = max(
                data_available(task_name, node_name), node_available(node_name)
            )
            exec_time = execution_time[task_name, node_name]
            new_task = ScheduledTask(
                node=node_name,
                name=task_name,
                start=start_time,
                end=start_time + exec_time,
            )
            scheduled_tasks[task_name] = new_task
            comp_schedule.add_task(new_task)

            clear_caches()

        return comp_schedule
