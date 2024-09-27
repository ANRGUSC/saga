from functools import partial
from typing import Dict, Hashable, List

import networkx as nx

from ..scheduler import Scheduler, Task


class MCTScheduler(Scheduler): # pylint: disable=too-few-public-methods
    """Minimum Completion Time scheduler
    
    Source: https://doi.org/10.1006/jpdc.2000.1714
    """
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Returns the schedule of the tasks on the network

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: The schedule of the tasks on the network.
        """
        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}  # Initialize list for each node
        scheduled_tasks: Dict[Hashable, Task] = {} # Map from task_name to Task

        def get_exec_time(task: Hashable, node: Hashable) -> float:
            return task_graph.nodes[task]['weight'] / network.nodes[node]['weight']

        def get_commtime(task1: Hashable, task2: Hashable, node1: Hashable, node2: Hashable) -> float:
            return task_graph.edges[task1, task2]['weight'] / network.edges[node1, node2]['weight']

        def get_eat(node: Hashable) -> float:
            eat = schedule[node][-1].end if schedule.get(node) else 0
            return eat

        def get_fat(task: Hashable, node: Hashable) -> float:
            fat = 0 if task_graph.in_degree(task) <= 0 else max(
                scheduled_tasks[pred_task].end +
                get_commtime(pred_task, task, scheduled_tasks[pred_task].node, node)
                for pred_task in task_graph.predecessors(task)
            )
            return fat

        def get_completion_time(task: Hashable, node: Hashable) -> float:
            start_time = max(get_eat(node), get_fat(task, node))
            return start_time + get_exec_time(task, node)

        for task in nx.topological_sort(task_graph):
            # Find node with minimum execution time for the task
            sched_node = min(network.nodes, key=partial(get_completion_time, task))

            start_time = max(get_eat(sched_node), get_fat(task, sched_node))
            end_time = start_time + get_exec_time(task, sched_node)

            # Add task to the schedule
            new_task = Task(node=sched_node, name=task, start=start_time, end=end_time)
            schedule[sched_node].append(new_task)
            scheduled_tasks[task] = new_task
        return schedule
