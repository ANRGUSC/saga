from queue import PriorityQueue
from typing import Dict, Hashable, List, Tuple
import networkx as nx
from saga.scheduler import Task
import random
from .tie_breaker import TieBreaker
class RandomTieBreaker(TieBreaker):
    def __init__(self):
        pass
    def __call__(
        self,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        runtimes: Dict[Hashable, Dict[Hashable, float]],
        commtimes: Dict[
            Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]
        ],
        comp_schedule: Dict[Hashable, List[Task]],
        task_schedule: Dict[Hashable, Task],
        priority_queue: List,
    ) -> Tuple[Hashable, int]:
        return random.choice(priority_queue)
