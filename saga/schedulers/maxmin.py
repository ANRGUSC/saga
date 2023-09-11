from functools import lru_cache
from typing import Dict, Hashable, List
import networkx as nx

from ..scheduler import Scheduler, Task


class MaxMinScheduler(Scheduler): # pylint: disable=too-few-public-methods
    """Max-Min scheduler"""
    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Schedules the task graph on the network

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: The schedule.
        """
        schedule: Dict[Hashable, List[Task]] = {}
        scheduled_tasks: Dict[Hashable, Task] = {} # Map from task_name to Task

        @lru_cache(maxsize=None)
        def get_eet(task: Hashable, node: Hashable) -> float:
            return task_graph.nodes[task]['weight'] / network.nodes[node]['weight']

        @lru_cache(maxsize=None)
        def get_commtime(task1: Hashable, task2: Hashable, node1: Hashable, node2: Hashable) -> float:
            return task_graph.edges[task1, task2]['weight'] / network.edges[node1, node2]['weight']

        @lru_cache(maxsize=None) # Must clear cache after each iteration since schedule changes
        def get_eat(node: Hashable) -> float:
            return schedule[node][-1].end if schedule.get(node) else 0

        @lru_cache(maxsize=None) # Must clear cache after each iteration since schedule changes
        def get_fat(task: Hashable, node: Hashable) -> float:
            return 0 if task_graph.in_degree(task) <= 0 else max([
                scheduled_tasks[pred_task].end +
                get_commtime(pred_task, task, scheduled_tasks[pred_task].node, node)
                for pred_task in task_graph.predecessors(task)
            ])

        @lru_cache(maxsize=None) # Must clear cache after each iteration since schedule changes
        def get_ect(task: Hashable, node: Hashable) -> float:
            return get_eet(task, node) + max(get_eat(node), get_fat(task, node))

        def clear_caches():
            """Clear all caches."""
            get_eat.cache_clear()
            get_fat.cache_clear()
            get_ect.cache_clear()

        while len(scheduled_tasks) < task_graph.order():
            available_tasks = [
                task for task in task_graph.nodes
                if task not in scheduled_tasks and all(pred in scheduled_tasks
                                                       for pred in task_graph.predecessors(task))
            ]
            #slight change from MinMin logic
            while available_tasks:
                min_ects = {task: min(get_ect(task, node) for node in network.nodes) for task in available_tasks}
                sched_task = max(min_ects, key=min_ects.get)
                sched_node = min(network.nodes, key=lambda node: get_ect(sched_task, node))

                schedule.setdefault(sched_node, [])
                new_task = Task(
                    node=sched_node,
                    name=sched_task,
                    start=max(get_eat(sched_node), get_fat(sched_task, sched_node)),
                    end=get_ect(sched_task, sched_node)
                )
                schedule[sched_node].append(new_task)
                scheduled_tasks[sched_task] = new_task

                available_tasks.remove(sched_task)

                clear_caches()

        return schedule
