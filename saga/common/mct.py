from itertools import product
from typing import Dict, Hashable, List

import networkx as nx

from ..base import Scheduler, Task


class MCTScheduler(Scheduler):
    """Minimum Completion Time scheduler"""
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Returns the schedule of the tasks on the network

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: The schedule of the tasks on the network.
        """
        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}  # Initialize list for each node

        def get_completion_time(task: Hashable, node: Hashable) -> float:
            exec_time = task_graph.nodes[task]['weight'] / network.nodes[node]['weight']
            last_task_end = schedule[node][-1].end if schedule.get(node) else 0
            return last_task_end + exec_time

        for task in task_graph.nodes:
            # Find node with minimum completion time for the task
            sched_task, sched_node = min(
                product([task], network.nodes),
                key=lambda instance: get_completion_time(instance[0], instance[1])
            )

            # Calculate start and end times
            start_time = schedule[sched_node][-1].end if schedule[sched_node] else 0
            end_time = start_time + task_graph.nodes[task]['weight'] / network.nodes[sched_node]['weight']

            # Add task to the schedule
            schedule[sched_node].append(Task(node=sched_node, name=sched_task, start=start_time, end=end_time))

        return schedule
