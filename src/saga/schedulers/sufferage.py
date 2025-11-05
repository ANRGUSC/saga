from functools import lru_cache
from typing import Dict, Hashable, List
import networkx as nx
from saga.utils.online_tools import ScheduleInjector

from ..scheduler import Scheduler, ScheduledTask

class SufferageScheduler(Scheduler):
    """Implements a sufferage scheduler.
    
    Source: https://dx.doi.org/10.1007/978-3-540-69277-5_7
    """
    def schedule(
        self,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        comp_schedule: Dict[Hashable, List[ScheduledTask]],
        task_map: Dict[Hashable, ScheduledTask],
        current_moment: float,
        **algo_kwargs
    ) -> Dict[Hashable, List[ScheduledTask]]:
        """Schedules the task graph on the network

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: The schedule.
        """
        # schedule: Dict[Hashable, List[Task]] = {}
        # scheduled_tasks: Dict[Hashable, Task] = {}  # Map from task_name to Task

        def get_eet(task: Hashable, node: Hashable) -> float:
            """Estimated execution time of a task on a node"""
            return task_graph.nodes[task]['weight'] / network.nodes[node]['weight']

        def get_commtime(task1: Hashable, task2: Hashable, node1: Hashable, node2: Hashable) -> float:
            """Communication time to send task1's output from node1 to task2's input on node2"""
            return task_graph.edges[task1, task2]['weight'] / network.edges[node1, node2]['weight']

        def get_eat(node: Hashable) -> float:
            """Earliest available time on a node"""
            return comp_schedule[node][-1].end if comp_schedule.get(node) else 0

        def get_fat(task: Hashable, node: Hashable) -> float:
            """Get file availability time of a task on a node"""
            return 0 if task_graph.in_degree(task) <= 0 else max([
                task_map[pred_task].end +
                get_commtime(pred_task, task, task_map[pred_task].node, node)
                for pred_task in task_graph.predecessors(task)
            ])
            
        def get_ect(task: Hashable, node: Hashable) -> float:
            """Get estimated completion time of a task on a node"""
            return get_eet(task, node) + max(get_eat(node), get_fat(task, node))

        while len(task_map) < task_graph.order():
            available_tasks = [
                task for task in task_graph.nodes
                if task not in task_map and all(pred in task_map
                    for pred in task_graph.predecessors(task))
            ]

            sufferages = {}
            for task in available_tasks:
                ect_values = [get_ect(task, node) for node in network.nodes]
                first_ect = min(ect_values)
                ect_values.remove(first_ect)
                second_ect = min(ect_values) if ect_values else first_ect
                sufferages[task] = second_ect - first_ect

            sched_task = max(sufferages, key=sufferages.get)
            sched_node = min(network.nodes, key=lambda node: get_ect(sched_task, node))

            comp_schedule.setdefault(sched_node, [])
            new_task = ScheduledTask(
                node=sched_node,
                name=sched_task,
                start=max(get_eat(sched_node), get_fat(sched_task, sched_node)),
                end=get_ect(sched_task, sched_node)
            )
            comp_schedule[sched_node].append(new_task)
            task_map[sched_task] = new_task
        return comp_schedule
