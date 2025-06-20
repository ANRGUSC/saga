from typing import Dict, Hashable, List, Optional, Tuple, Set

import networkx as nx
import numpy as np
from copy import deepcopy

from saga.scheduler import Task

from ..scheduler import Task
from ..scheduler import Scheduler


class ETFScheduler(Scheduler): # pylint: disable=too-few-public-methods
    """Earliest Task First scheduler"""

    def _get_start_times(self,
                         task_map: Dict[Hashable, Task],
                         ready_tasks: Set[Hashable],
                         ready_nodes: Set[Hashable],
                         task_graph: nx.DiGraph,
                         network: nx.Graph,
                         min_start_time: float = 0.0,) -> Dict[Hashable, Tuple[Hashable, float]]:
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
            mini_min_start_time, min_node = np.inf, None #renamed the local min_start_time to mini_min_start time for consistancy 
            for node in ready_nodes:
                max_arrival_time = max([
                    min_start_time, *[ #added min_start_time
                        task_map[parent].end + (
                            task_graph.edges[parent, task]["weight"] /
                            network.edges[task_map[parent].node, node]["weight"]
                        ) for parent in task_graph.predecessors(task)
                    ]
                ])
                if max_arrival_time < mini_min_start_time:
                    mini_min_start_time = max_arrival_time
                    min_node = node
            start_times[task] = min_node, mini_min_start_time
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

    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 #clusters: Optional[List[Set[Hashable]]] = None,
                 schedule: Optional[Dict[str, List[Task]]] = None, #new
                 min_start_time: float = 0.0) -> Dict[str, List[Task]]: #new
        """Returns the best schedule (minimizing makespan) for a problem instance using ETF

        Args:
            network: Network
            task_graph: Task graph
            

        Returns:
            A dictionary of the schedule
        """
        current_moment = min_start_time #changed from zero. This is the "internal clock" for the algorithm
        next_moment = np.inf

        #new
        comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes} #changed name for consistancy 
        task_map: Dict[Hashable, Task] = {} #changed from tasks to task_map for consistancy 

        print("testing")
        print(schedule)
        if schedule is not None:
            # start from tasks already executed
            comp_schedule = deepcopy(schedule)
            task_map = {t.name: t for tasks in comp_schedule.values() for t in tasks}
            if task_map:
                # set clock to end of last executed task
                current_moment = min_start_time#min(t.end for tasks in comp_schedule.values() for t in tasks)
                # find next completion after current_moment
                next_moment = min((t.end for tasks in comp_schedule.values() for t in tasks if t.end > current_moment), default=np.inf)
            else:
                current_moment = min_start_time
                next_moment = np.inf
        else:
            # initialize fresh schedule
            comp_schedule = {node: [] for node in network.nodes}
            task_map = {}
            current_moment = min_start_time
            next_moment = np.inf
        


        while len(task_map) < len(task_graph.nodes): # While there are still tasks to schedule
            ready_tasks = self._get_ready_tasks(task_map, task_graph)
            ready_nodes = {
                node for node in network.nodes
                if not comp_schedule[node] or comp_schedule[node][-1].end <= current_moment
            }
            while ready_tasks and ready_nodes:
                start_times = self._get_start_times(task_map, ready_tasks, ready_nodes, task_graph, network, min_start_time=current_moment) #added min_start_time
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
                    comp_schedule[node_to_schedule_on].append(new_task)
                    task_map[task_to_schedule] = new_task
                    ready_tasks.remove(task_to_schedule)
                    ready_nodes.remove(node_to_schedule_on)
                    if new_task.end < next_moment:
                        next_moment = new_task.end
                else:
                    break

            current_moment = next_moment
            next_moment = min([np.inf, *[task.end for task in task_map.values() if task.end > current_moment]])

        return comp_schedule
