from typing import Dict, Hashable, List, Optional
import networkx as nx
import numpy as np
import heapq
from queue import Queue

from ..base import Scheduler, Task
from ..utils.tools import check_instance_simple

# class OLBScheduler(Scheduler):
#     def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
#         # Initialize a priority queue (heap) for each processor in the network
#         processors = {node: [] for node in network.nodes}
        
#         # Assign tasks to processors
#         tasks = list(task_graph.nodes)
#         for i, task in enumerate(tasks):
#             # Get the processor with the least number of tasks
#             min_processor = min(processors, key=lambda x: len(processors[x]))
#             # Create a Task object and add it to the processor's queue
#             processors[min_processor].append(Task(min_processor, task))
        
#         return processors

class OLBScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        check_instance_simple(network, task_graph)
        schedule = {node: [] for node in network.nodes}

        # Queue tasks
        task_queue = Queue()
        for task in task_graph.nodes:
            task_queue.put(task)

        # Schedule tasks on available nodes
        while not task_queue.empty():
            for node in schedule.keys():
                if not task_queue.empty():
                    task = task_queue.get()
                    schedule[node].append(Task(node=node, name=task))

        return schedule
