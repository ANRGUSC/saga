from abc import ABC, abstractmethod
import networkx as nx
from typing import Dict, Hashable, List, Tuple
from saga.scheduler import Task
class InsertTask(ABC):
    @abstractmethod
    def __call__(
        self,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        runtimes: Dict[Hashable, Dict[Hashable, float]],
        commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
        comp_schedule: Dict[Hashable, List[Task]],
        task_schedule: Dict[Hashable, Task],
        task_name: Hashable,
        priority: int,
        rankings:List
        ) -> None:  
        raise NotImplementedError