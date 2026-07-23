from functools import lru_cache
from itertools import product
from typing import Dict, Hashable, List, Tuple

import networkx as nx

from ....scheduler import Scheduler, Task
from saga.utils.draw import draw_gantt


class NPMinMinScheduler(Scheduler):
    """Minimum Completion Time scheduler"""
    def schedule(self, network: nx.Graph, task_graphs: List[Tuple[nx.DiGraph, float]]) -> Dict[Hashable, List[Task]]:
        """Returns the schedule of the tasks on the network

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: The schedule of the tasks on the network.
        """
        schedule: Dict[Hashable, List[Task]] = {}
        scheduled_tasks: Dict[Hashable, Task] = {} # Map from task_name to Task

        for task_graph_tupple in task_graphs:
            task_graph = task_graph_tupple[0]
            task_graph_arrival_time = task_graph_tupple[1]

            @lru_cache(maxsize=None)
            def get_eet(task: Hashable, node: Hashable) -> float:
                return task_graph.nodes[task]['weight'] / network.nodes[node]['weight']

            @lru_cache(maxsize=None)
            def get_commtime(task1: Hashable, task2: Hashable, node1: Hashable, node2: Hashable) -> float:
                return task_graph.edges[task1, task2]['weight'] / network.edges[node1, node2]['weight']


            # this function gives the earliest available time of a node which is the end time of the last task scheduled on that node or 0 if no task is scheduled on that node
            @lru_cache(maxsize=None) # Must clear cache after each iteration since schedule changes
            def get_eat(node: Hashable) -> float:
                eat = schedule[node][-1].end if schedule.get(node) else task_graph_arrival_time
                return eat


            # this function gives the latest available time of a task on a node which is the maximum of the end time of the tasks that are predecessors of the task and the communication time between the predecessor task and the task
            @lru_cache(maxsize=None) # Must clear cache after each iteration since schedule changes
            def get_fat(task: Hashable, node: Hashable) -> float:
                fat = task_graph_arrival_time if task_graph.in_degree(task) <= 0 else max([
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

            # while len(scheduled_tasks) < task_graph.order():
            while set(task_graph.nodes).issubset(set(scheduled_tasks.keys())) == False:
                available_tasks = [
                    task for task in task_graph.nodes
                    if (task not in scheduled_tasks and
                        # Check if all predecessors are scheduled
                        set(task_graph.predecessors(task)).issubset(set(scheduled_tasks.keys()))) 
                ]
                while available_tasks:
                    # Find the task and node that minimizes the ECT
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

           
        # add empty list for nodes that have no tasks scheduled
        for node in network.nodes:
            schedule.setdefault(node, [])
        
        # sort schedule pairs by key name
        schedule = dict(sorted(schedule.items()))

        return schedule
