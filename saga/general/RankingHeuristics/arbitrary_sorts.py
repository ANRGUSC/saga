import random
import networkx as nx
from queue import PriorityQueue

def random_rank_sort(_: nx.Graph, task_graph: nx.DiGraph) -> PriorityQueue:
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
