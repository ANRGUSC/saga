from abc import ABC, abstractmethod
import networkx as nx
from typing import Dict, Hashable, List, Tuple
from saga.scheduler import Task

class Filter(ABC):
    @abstractmethod
    def __call__(
        self,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        priority_queue: List,
        schedule: Dict[Hashable, List[Task]],
    ):
        raise NotImplementedError