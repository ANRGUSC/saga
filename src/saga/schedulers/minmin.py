from functools import lru_cache
from itertools import product
from typing import Dict, Hashable, List

import networkx as nx

from ..scheduler import Scheduler, Task


class MinMinScheduler(Scheduler):
    """Minimum Completion Time scheduler"""
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Returns the schedule of the tasks on the network

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: The schedule of the tasks on the network.
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
            eat = schedule[node][-1].end if schedule.get(node) else 0
            return eat

        @lru_cache(maxsize=None) # Must clear cache after each iteration since schedule changes
        def get_fat(task: Hashable, node: Hashable) -> float:
            fat = 0 if task_graph.in_degree(task) <= 0 else max([
                scheduled_tasks[pred_task].end +
                get_commtime(pred_task, task, scheduled_tasks[pred_task].node, node)
                for pred_task in task_graph.predecessors(task)
            ])
            return fat

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
                if (task not in scheduled_tasks and
                    set(task_graph.predecessors(task)).issubset(set(scheduled_tasks.keys())))
            ]
            while available_tasks:
                sched_task, sched_node = min(
                    product(available_tasks, network.nodes),
                    key=lambda instance: get_ect(instance[0], instance[1])
                )
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
