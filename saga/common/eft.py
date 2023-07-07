import logging
import pathlib
from typing import Dict, Hashable, List, Tuple, Set

import networkx as nx
import numpy as np

from saga.base import Task

from ..base import Task
from ..base import Scheduler

class EFTScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__()

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        current_moment = 0
        next_moment = np.inf

        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        tasks: Dict[Hashable, Task] = {}

        def get_ready_tasks() -> Set[Hashable]:
            return {
                task for task in task_graph.nodes
                if task not in tasks and all(pred in tasks for pred in task_graph.predecessors(task))
            }
        
        while len(tasks) < len(task_graph.nodes): # While there are still tasks to schedule
            ready_tasks = get_ready_tasks()
            ready_nodes = {
                node for node in network.nodes
                if not schedule[node] or schedule[node][-1].end <= current_moment
            }
            while ready_tasks and ready_nodes:
                start_times: Dict[Hashable, Tuple[Hashable, float]] = {}
                for task in ready_tasks:
                    min_start_time, min_node = np.inf, None
                    for node in ready_nodes:
                        max_arrival_time = max([
                            0.0, *[
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
                    
                task_to_schedule = min(list(start_times.keys()), key=lambda task: start_times[task][1])
                node_to_schedule_on, start_time = start_times[task_to_schedule]

                start_time = max(start_time, current_moment)

                if start_time <= next_moment:
                    new_task = Task(
                        node=node_to_schedule_on,
                        name=task_to_schedule,
                        start=start_time,
                        end=start_time + task_graph.nodes[task_to_schedule]["weight"] / network.nodes[node_to_schedule_on]["weight"]
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
            
                




