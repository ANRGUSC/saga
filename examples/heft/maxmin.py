from saga.scheduler import Scheduler, Task

import itertools
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np


class MaxMinScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Returns the best schedule (minimizing makespan) for a problem instance using Minmin

        Args:
            network: Network
            task_graph: Task graph

        Returns:
            A dictionary of the schedule
        """
        # Initialize schedule as an empty dictionary
        schedule = {node: [] for node in network.nodes}

        # While there are unscheduled tasks
        while task_graph.number_of_nodes() > 0:
            # Calculate earliest completion time for each task on each machine
            ect = {}
            for task in task_graph.nodes:
                for machine in network.nodes:
                    communication_time = max(
                        [task_graph.edges[pred, task]['weight'] for pred in task_graph.predecessors(task)] or [0])
                    execution_time = task_graph.nodes[task]['weight'] / \
                        network.nodes[machine]['weight']
                    completion_time = max([task.end for task in schedule[machine]] or [
                                          0]) + communication_time + execution_time
                    ect[(task, machine)] = completion_time

            # Find task-machine pair with minimum earliest completion time
            max_machine, max_task = min(ect)

            # Add the task to the schedule
            schedule[max_machine].append(Task(max_machine, max_task, max(
                [task.end for task in schedule[max_machine]] or [0]), ect[(max_task, max_machine)]))
            

            # Remove the scheduled task from the task graph
            task_graph.remove_node(max_task)

        return schedule
