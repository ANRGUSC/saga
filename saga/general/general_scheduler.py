import logging
from typing import Dict, Hashable, List, Tuple
import networkx as nx
from ..scheduler import Scheduler, Task
from utils import get_runtimes
class GeneralScheduler(Scheduler):
    def __init__(self, ranking_heuristic) -> None:
        self.ranking_heauristic = ranking_heuristic

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph
    ) -> Dict[Hashable, List[Task]]:
        rankings = self.ranking_heauristic(network, task_graph)
