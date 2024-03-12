from abc import ABC, abstractmethod
import networkx as nx
from typing import Dict, Hashable, List, Tuple
from saga.scheduler import Task
from .filter import Filter

class KFirstFilter(Filter):
    def __init__(self, k: int = 1):
        self.k = k
    def __call__(
        self,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        priority_queue: List,
        schedule: Dict[Hashable, List[Task]],
    ):
        return priority_queue[:self.k]