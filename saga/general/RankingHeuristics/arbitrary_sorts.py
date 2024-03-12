import random
import networkx as nx
from queue import PriorityQueue
from .ranking_heuristic import RankingHeuristic

class RandomRankSort(RankingHeuristic):
    def __init__(self):
        pass
    
    def __call__(self, _: nx.Graph, task_graph: nx.DiGraph) -> PriorityQueue:
        """
        Sorts the tasks in the task graph by their random rank.
        
        Args:
            network (nx.Graph): The network to schedule onto.
            task_graph (nx.DiGraph): The task graph to schedule.
        
        Returns:
            List[Hashable]: The sorted tasks.
        """

        queue = []
        for task_name in task_graph.nodes:
            queue.append((task_name, None))
        
        return sorted(queue, key=lambda _: random.random(), reverse=True)
