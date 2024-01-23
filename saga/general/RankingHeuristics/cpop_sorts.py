from typing import Dict, Hashable, List
from queue import PriorityQueue
from .ranking_heuristic import RankingHeuristic
import networkx as nx
import numpy as np


def upward_rank(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """Computes the upward rank of the tasks in the task graph."""
    ranks = {}

    is_comm_zero = all(np.isclose(network.edges[_edge]['weight'], 0) for _edge in network.edges)
    is_comp_zero = all(np.isclose(network.nodes[_node]['weight'], 0) for _node in network.nodes)

    def avg_comm_time(parent: Hashable, child: Hashable) -> float:
        if is_comm_zero:
            return 1e-9
        return np.mean([ # average communication time for output data of predecessor
            task_graph.edges[parent, child]['weight'] / network.edges[src, dst]['weight']
            for src, dst in network.edges
            if not np.isclose(network.edges[src, dst]['weight'], 0)
        ])

    def avg_comp_time(task: Hashable) -> float:
        if is_comp_zero:
            return 1e-9
        return np.mean([
            task_graph.nodes[task]['weight'] / network.nodes[node]['weight']
            for node in network.nodes
            if not np.isclose(network.nodes[node]['weight'], 0)
        ])

    for task_name in reversed(list(nx.topological_sort(task_graph))):
        max_comm = 0 if task_graph.out_degree(task_name) <= 0 else max(
            (
                ranks[succ] + # rank of successor
                avg_comm_time(task_name, succ) # average communication time for output data of task
            )
            for succ in task_graph.successors(task_name)
        )
        ranks[task_name] = avg_comp_time(task_name) + max_comm

    return ranks

def downward_rank(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """Computes the downward rank of the tasks in the task graph."""
    ranks = {}

    is_comm_zero = all(np.isclose(network.edges[_edge]['weight'], 0) for _edge in network.edges)
    is_comp_zero = all(np.isclose(network.nodes[_node]['weight'], 0) for _node in network.nodes)

    def avg_comm_time(parent: Hashable, child: Hashable) -> float:
        if is_comm_zero:
            return 1e-9
        return np.mean([ # average communication time for output data of predecessor
            task_graph.edges[parent, child]['weight'] / network.edges[src, dst]['weight']
            for src, dst in network.edges
            if not np.isclose(network.edges[src, dst]['weight'], 0)
        ])

    def avg_comp_time(task: Hashable) -> float:
        if is_comp_zero:
            return 1e-9
        return np.mean([
            task_graph.nodes[task]['weight'] / network.nodes[node]['weight']
            for node in network.nodes
            if not np.isclose(network.nodes[node]['weight'], 0)
        ])

    for task_name in nx.topological_sort(task_graph):
        ranks[task_name] = 0 if task_graph.in_degree(task_name) <= 0 else max(
            (
                avg_comp_time(pred) + ranks[pred] + avg_comm_time(pred, task_name)
            )
            for pred in task_graph.predecessors(task_name)
        )

    return ranks

class UpwardRankSort(RankingHeuristic):
    def __init__(self):
        pass
    def __call__(self, network: nx.Graph, task_graph: nx.DiGraph) -> List:
        """
        Sorts the tasks in the task graph by their upward rank.
        
        Args:
            network (nx.Graph): The network to schedule onto.
            task_graph (nx.DiGraph): The task graph to schedule.
        
        Returns:
            List[Hashable]: The sorted tasks.
        """
        rank = upward_rank(network, task_graph)
        queue = []
        for task_name in task_graph.nodes:
            queue.append((task_name, None))

        return sorted(queue, key=lambda x: rank[x[0]], reverse=True)

class DownwardRankSort(RankingHeuristic):
    def __init__(self):
        pass
    def __call__(self, network: nx.Graph, task_graph: nx.DiGraph) -> List:
        """
        Sorts the tasks in the task graph by their downward rank.
        
        Args:
            network (nx.Graph): The network to schedule onto.
            task_graph (nx.DiGraph): The task graph to schedule.
        
        Returns:
            List[Hashable]: The sorted tasks.
        """
        rank = downward_rank(network, task_graph)
        queue = []
        for task_name in task_graph.nodes:
            queue.append((task_name, None))

        return sorted(queue, key=lambda x: rank[x[0]], reverse=True)

class CriticalPathSort(RankingHeuristic):
    def __init__(self):
        pass

    def __call__(self, network: nx.Graph, task_graph: nx.DiGraph) -> List:
        """
        Sorts the tasks in the task graph by their upward+downward rank.
        
        Args:
            network (nx.Graph): The network to schedule onto.
            task_graph (nx.DiGraph): The task graph to schedule.
        
        Returns:
            List[Hashable]: The sorted tasks.
        """
        upward_ranks = upward_rank(network, task_graph)
        downward_ranks = downward_rank(network, task_graph)
        queue = []
        start_task = next(task for task in task_graph.nodes if task_graph.in_degree(task) == 0)
        cp_val = upward_ranks[start_task] + downward_ranks[start_task]
        
        for task_name in task_graph.nodes:
            priority = 1 if np.isclose(cp_val, upward_ranks[task_name] + downward_ranks[task_name]) else 0
            queue.append((task_name, priority))
        
        return sorted(queue, key=lambda x: upward_ranks[x[0]] + downward_ranks[x[0]], reverse=True)