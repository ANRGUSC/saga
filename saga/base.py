from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional
import networkx as nx

@dataclass
class Task:
    node: str
    name: str
    start: Optional[float]
    end: Optional[float] 

class Scheduler(ABC):
    @abstractmethod
    def __init__(self) -> None:
        super(Scheduler, self).__init__()

    @abstractmethod
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        pass
