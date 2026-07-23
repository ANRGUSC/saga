from functools import lru_cache, partial
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np

from ....scheduler import Scheduler, Task


class ResidualGDLScheduler(Scheduler):
    """Generalized Dynamic Level Scheduler

    Source: https://doi.org/10.1109/71.207593
    Notes: Considers homogenous communication speeds (not homogenous compute speeds, though)
    """
    def __init__(self, dynamic_level: int = 2):
        super().__init__()
        self.dynamic_level = dynamic_level
        if dynamic_level not in (1, 2):
            raise ValueError("dynamic_level must be 1 or 2")

    def schedule(self, network: nx.Graph, task_graphs: List[Tuple[nx.DiGraph, float]]) -> Dict[Hashable, List[Task]]:
        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        scheduled_tasks: Dict[Hashable, Task] = {}

        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        scheduled_tasks: Dict[Hashable, Task] = {}

        for task_graph_tupple in task_graphs:
            task_graph = task_graph_tupple[0]
            task_graph_arrival_time = task_graph_tupple[1]

            execution_time = {}
            for task in task_graph.nodes:
                for node in network.nodes:
                    execution_time[task, node] = (
                        task_graph.nodes[task]['weight'] / network.nodes[node]['weight']
                    )

            communication_time = {}
            for dep in task_graph.edges:
                for link in network.edges:
                    communication_time[dep, tuple(sorted(link))] = (
                        task_graph.edges[dep]['weight'] / network.edges[link]['weight']
                    )

            median_execution_time_per_task = {
                task: np.median([
                    execution_time[task, node]
                    for node in network.nodes
                ])
                for task in task_graph.nodes
            }

            median_execution_time = np.median([
                execution_time[task, node]
                for task in task_graph.nodes
                for node in network.nodes
            ])

            delta_execution_time = {}
            for task in task_graph.nodes:
                for node in network.nodes:
                    delta_execution_time[task, node] = (
                        median_execution_time - execution_time[task, node]
                    )

            # the static level of a task is the largest sum of execution times
            # along any directed path from the task to a sink
            static_level = {}
            for task in reversed(list(nx.topological_sort(task_graph))):
                if task_graph.out_degree(task) == 0:
                    static_level[task] = median_execution_time_per_task[task]
                else:
                    static_level[task] = median_execution_time_per_task[task] + max(
                        static_level[succ] + task_graph.edges[task, succ]['weight']
                        for succ in task_graph.successors(task)
                    )

            @lru_cache(maxsize=None)
            def data_available(task: str, node: str) -> float:
                """returns time when data is available to execute task on node"""
                if task_graph.in_degree(task) == 0:
                    return task_graph_arrival_time
                return max(
                    scheduled_tasks[pred].end + ( # finish time + communication time
                        task_graph.edges[pred, task]['weight'] /
                        network.edges[scheduled_tasks[pred].node, node]['weight']
                    )
                    for pred in task_graph.predecessors(task)
                )

            @lru_cache(maxsize=None)
            def node_available(node: str) -> float:
                """returns time when node is available to execute task"""
                if len(schedule[node]) == 0:
                    return 0
                return max(task.end for task in schedule[node])

            @lru_cache(maxsize=None)
            def dynamic_level_1(task: str, node: str) -> float:
                return static_level[task] - max(
                    data_available(task, node),
                    node_available(node)
                ) + delta_execution_time[task, node]

            largest_output_descendants = {
                task: max(
                    task_graph.successors(task),
                    key=partial(
                        lambda t, s: task_graph.edges[t, s]['weight'],
                        task
                    )
                )
                for task in task_graph.nodes
                if task_graph.out_degree(task) > 0
            }

            @lru_cache(maxsize=None)
            def descendant_earliest_finish(task, child, node):
                """returns earliest finish time of child's descendants on node

                NOTE: slight difference from paper, comm time is *inside* the
                min function since we have heterogeneous comm speeds. This is
                backwards-compatible with the paper's definition when the comm
                speeds are homogeneous.
                """
                if len(network.nodes) == 1:
                    return np.inf
                return min(
                    communication_time[(task, child), tuple(sorted((node, other)))] +
                    execution_time[child, other]
                    for other in network.nodes
                    if node != other
                )

            @lru_cache(maxsize=None)
            def descendant_consideration(task: str, node: str) -> float:
                dc = largest_output_descendants.get(task) # pylint: disable=invalid-name
                if dc is None:
                    return 0
                return median_execution_time_per_task[dc] - min(
                    execution_time[dc, node],
                    descendant_earliest_finish(task, dc, node)
                )

            @lru_cache(maxsize=None)
            def dynamic_level_2(task: str, node: str) -> float:
                return dynamic_level_1(task, node) + descendant_consideration(task, node)

            @lru_cache(maxsize=None)
            def preferred_node(task: str) -> str:
                return min(
                    network.nodes,
                    key=lambda node: dynamic_level(task, node)
                )

            dynamic_level = dynamic_level_1 if self.dynamic_level == 1 else dynamic_level_2
            @lru_cache(maxsize=None)
            def cost(task: str) -> float:
                return dynamic_level(task, preferred_node(task)) - max(
                    dynamic_level(task, other)
                    for other in network.nodes
                )

            @lru_cache(maxsize=None)
            def global_dynamic_level(task: str) -> float:
                return dynamic_level(task, preferred_node(task)) + cost(task)

            def clear_caches():
                """Clears the scheduler's caches"""
                data_available.cache_clear()
                node_available.cache_clear()
                dynamic_level_1.cache_clear()
                dynamic_level_2.cache_clear()
                descendant_earliest_finish.cache_clear()
                descendant_consideration.cache_clear()
                dynamic_level.cache_clear()
                cost.cache_clear()
                global_dynamic_level.cache_clear()

            # while len(scheduled_tasks) < len(task_graph.nodes):
            while set(task_graph.nodes).issubset(set(scheduled_tasks.keys())) == False:
                candidate_tasks = [
                    task for task in task_graph.nodes
                    if task not in scheduled_tasks and all(
                        pred in scheduled_tasks
                        for pred in task_graph.predecessors(task)
                    )
                ]
                task = min(candidate_tasks, key=global_dynamic_level)
                node = preferred_node(task)
                start_time = max(
                    data_available(task, node),
                    node_available(node)
                )
                exec_time = execution_time[task, node]
                new_task = Task(
                    node=node,
                    name=task,
                    start=start_time,
                    end=start_time+exec_time
                )
                scheduled_tasks[task] = new_task
                schedule[node].append(new_task)

                clear_caches()

            return schedule
