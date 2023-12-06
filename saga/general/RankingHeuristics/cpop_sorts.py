from typing import Dict, Hashable, List
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


def upward_rank_sort(network: nx.Graph, task_graph: nx.DiGraph) -> List[Hashable]:
    rank = upward_rank(network, task_graph)

    return sorted(list(rank.keys()), key=rank.get, reverse=True)

def downward_rank_sort(network: nx.Graph, task_graph: nx.DiGraph) -> List[Hashable]:
    rank = downward_rank(network, task_graph)

    return sorted(list(rank.keys()), key=rank.get, reverse=True)

def upward_downward_rank(network: nx.Graph, task_graph: nx.DiGraph) -> List[Hashable]:
    upward_ranks = upward_rank(network, task_graph)
    downward_ranks = downward_rank(network, task_graph)
    return {
        task_name: (upward_ranks[task_name] + downward_ranks[task_name])
        for task_name in task_graph.nodes
    }