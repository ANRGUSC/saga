from typing import Dict, Hashable, List, Tuple, Set

import networkx as nx
import numpy as np

from saga.scheduler import Task

from ....scheduler import Task
from ....scheduler import Scheduler


class ResidualETFScheduler(Scheduler): # pylint: disable=too-few-public-methods
    """Earliest Task First scheduler"""

    def _get_start_times(self,
                         tasks: Dict[Hashable, Task],
                         ready_tasks: Set[Hashable],
                         ready_nodes: Set[Hashable],
                         task_graph: nx.DiGraph,
                         task_graph_arrival_time: float,
                         network: nx.Graph) -> Dict[Hashable, Tuple[Hashable, float]]:
        """Returns the earliest possible start times of the ready tasks on the ready nodes
  

        Args:
            tasks (Dict[Hashable, Task]): The tasks.
            ready_tasks (Set[Hashable]): The ready tasks.
            ready_nodes (Set[Hashable]): The ready nodes.
            task_graph (nx.DiGraph): The task graph.
            network (nx.Graph): The network.

        Returns:
            Dict[Hashable, Tuple[Hashable, float]]: The start times of the ready tasks on the ready nodes.
        """
        start_times = {}
        for task in ready_tasks:
            min_start_time, min_node = np.inf, None
            for node in ready_nodes:
                max_arrival_time = max([
                    task_graph_arrival_time, *[
                        tasks[parent].end + (
                            task_graph.edges[parent, task]["weight"] /
                            network.edges[tasks[parent].node, node]["weight"]
                        ) for parent in task_graph.predecessors(task)
                    ]
                ])
                if max_arrival_time < min_start_time:
                    min_start_time = max_arrival_time
                    min_node = node
            start_times[task] = min_node, min_start_time
        return start_times

    def _get_ready_tasks(self, tasks: Dict[Hashable, Task], task_graph: nx.DiGraph) -> Set[Hashable]:
        """Returns the ready tasks

        Args:
            tasks (Dict[Hashable, Task]): The tasks.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Set[Hashable]: The ready tasks.
        """
        return {
            task for task in task_graph.nodes
            if task not in tasks and all(pred in tasks for pred in task_graph.predecessors(task))
        }
 

    def schedule(self, network: nx.Graph, task_graphs: List[Tuple[nx.DiGraph, float]]) -> Dict[Hashable, List[Task]]:
        """Returns the best schedule (minimizing makespan) for a problem instance using ETF

        Args:
            network: Network
            task_graph: Task graph

        Returns:
            A dictionary of the schedule
        """
        current_moment = 0
        next_moment = np.inf

        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        tasks: Dict[Hashable, Task] = {}


        for task_graph_tupple in task_graphs:
            task_graph = task_graph_tupple[0]
            task_graph_arrival_time = task_graph_tupple[1]


            # while len(tasks) < len(task_graph.nodes): 
            while set(task_graph.nodes).issubset(set(tasks.keys())) == False:
                ready_tasks = self._get_ready_tasks(tasks, task_graph)
                ready_nodes = {
                    node for node in network.nodes
                    if not schedule[node] or schedule[node][-1].end <= current_moment
                }
                while ready_tasks and ready_nodes:
                    start_times = self._get_start_times(tasks, ready_tasks, ready_nodes, task_graph, task_graph_arrival_time, network)
                    task_to_schedule = min(list(start_times.keys()), key=lambda task: start_times[task][1])
                    node_to_schedule_on, start_time = start_times[task_to_schedule]

                    start_time = max(start_time, current_moment)

                    if start_time <= next_moment:
                        new_task = Task(
                            node=node_to_schedule_on,
                            name=task_to_schedule,
                            start=start_time,
                            end=start_time + (
                                task_graph.nodes[task_to_schedule]["weight"] /
                                network.nodes[node_to_schedule_on]["weight"]
                            )
                        )
                        schedule[node_to_schedule_on].append(new_task)
                        tasks[task_to_schedule] = new_task
                        ready_tasks.remove(task_to_schedule)
                        ready_nodes.remove(node_to_schedule_on)
                        if new_task.end < next_moment:
                            next_moment = new_task.end
                    else:
                        break

                current_moment = next_moment
                next_moment = min([np.inf, *[task.end for task in tasks.values() if task.end > current_moment]])

        return schedule
    