from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Dict, Hashable, List, Optional
import networkx as nx
import numpy as np
import heapq

#from ..base import Scheduler, Task
#from ..utils.tools import check_instance_simple

@dataclass
class Task:
    def __init__(self, node: str, name: str, start: Optional[float] = None, end: Optional[float] = None):
        self.node = node
        self.name = name
        self.start = start
        self.end = end

class Scheduler(ABC):
    @abstractmethod
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        pass

class OLBScheduler(Scheduler):
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        # Initialize a priority queue (heap) for each processor in the network
        processors = {node: [] for node in network.nodes}
        
        # Assign tasks to processors
        tasks = list(task_graph.nodes)
        for i, task in enumerate(tasks):
            # Get the processor with the least number of tasks
            min_processor = min(processors, key=lambda x: len(processors[x]))
            # Create a Task object and add it to the processor's queue
            processors[min_processor].append(Task(min_processor, task))
        
        return processors
