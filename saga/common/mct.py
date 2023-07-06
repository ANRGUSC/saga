from typing import Dict, List, Optional, Hashable
import networkx as nx
import numpy as np
from itertools import product

from ..base import Scheduler, Task
from ..utils.tools import check_instance_simple

# class MCTScheduler:
#     def __init__(self, runtimes, machine_capacities):
#         self.runtimes = runtimes
#         self.machine_capacities = machine_capacities

#     def schedule(self, network, task_graph):
#         check_instance_simple(network, task_graph)
#         nodes = list(network.nodes())
#         processors = [0]*len(nodes)
#         schedule = {node: [] for node in nodes}

#         for task_node in nx.topological_sort(task_graph):
#             # Find the processor that finishes earliest
#             min_processor_index = np.argmin(processors)
#             min_processor = nodes[min_processor_index]
            
#             # Get task details
#             task_time = self.runtimes[task_node][task_node]
            
#             # Create a new task with the start time equal to the current finish time of the processor
#             task = Task(task_node, task_time, processors[min_processor_index], processors[min_processor_index] + task_time)
            
#             # Schedule the task on this processor and update the finish time
#             schedule[min_processor].append(task)
#             processors[min_processor_index] += task_time

#         return schedule

class MCTScheduler(Scheduler):
    def __init__(self):
        super(MCTScheduler, self).__init__()

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        check_instance_simple(network, task_graph)

        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}  # Initialize list for each node

        def CompletionTime(task: Hashable, node: Hashable) -> float:
            exec_time = task_graph.nodes[task]['weight'] / network.nodes[node]['weight']
            last_task_end = schedule[node][-1].end if schedule.get(node) else 0
            return last_task_end + exec_time
        
        for task in task_graph.nodes:
            # Find node with minimum completion time for the task
            sched_task, sched_node = min(
                product([task], network.nodes),
                key=lambda instance: CompletionTime(instance[0], instance[1])
            )
            
            # Calculate start and end times
            start_time = schedule[sched_node][-1].end if schedule[sched_node] else 0
            end_time = start_time + task_graph.nodes[task]['weight'] / network.nodes[sched_node]['weight']

            # Add task to the schedule
            schedule[sched_node].append(Task(node=sched_node, name=sched_task, start=start_time, end=end_time))

        return schedule



