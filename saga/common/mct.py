from typing import Dict, List, Optional, Hashable
import networkx as nx
import numpy as np

from ..base import Scheduler, Task
from ..utils.tools import check_instance_simple

class MCTScheduler:
    def __init__(self, runtimes, machine_capacities):
        self.runtimes = runtimes
        self.machine_capacities = machine_capacities

    def schedule(self, network, task_graph):
        check_instance_simple(network, task_graph)
        nodes = list(network.nodes())
        processors = [0]*len(nodes)
        schedule = {node: [] for node in nodes}

        for task_node in nx.topological_sort(task_graph):
            # Find the processor that finishes earliest
            min_processor_index = np.argmin(processors)
            min_processor = nodes[min_processor_index]
            
            # Get task details
            task_time = self.runtimes[task_node][task_node]
            
            # Create a new task with the start time equal to the current finish time of the processor
            task = Task(task_node, task_time, processors[min_processor_index], processors[min_processor_index] + task_time)
            
            # Schedule the task on this processor and update the finish time
            schedule[min_processor].append(task)
            processors[min_processor_index] += task_time

        return schedule




