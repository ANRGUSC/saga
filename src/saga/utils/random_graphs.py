from copy import deepcopy
from itertools import product
import random
from typing import Tuple, TypeVar

import networkx as nx
import numpy as np
from scipy.stats import norm

from saga.utils.random_variable import RandomVariable



def add_ccr_weights(task_graph: nx.DiGraph, 
                    network: nx.Graph,
                    ccr: float) -> nx.Graph:
    """Get the network with weights for a given CCR

    Args:
        task_graph (nx.DiGraph): The task graph.
        network (nx.Graph): The network graph.
        ccr (float): The communication to computation ratio.

    Returns:
        nx.Graph: The network graph.
    """
    network = deepcopy(network)
    mean_network_weight = np.mean([
        network.nodes[node]["weight"]
        for node in network.nodes
    ])
    mean_task_cost = np.mean([
        task_graph.nodes[node]["weight"]
        for node in task_graph.nodes
    ])
    mean_dependency_weight = np.mean([
        task_graph.edges[edge]["weight"]
        for edge in task_graph.edges
    ])
    
    link_strength = (mean_dependency_weight * mean_network_weight) / (ccr * mean_task_cost)

    for edge in network.edges:
        if edge[0] == edge[1]:
            network.edges[edge]["weight"] = 1e9
        else:
            network.edges[edge]["weight"] = link_strength

    return network

def  get_diamond_dag() -> nx.DiGraph:
    """Returns a diamond DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C", "D"])
    dag.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    return dag

def get_chain_dag(num_nodes: int = 4) -> nx.DiGraph:
    """Returns a chain DAG."""
    dag = nx.DiGraph()
    nodes = [chr(ord("A") + i) for i in range(num_nodes)]
    dag.add_nodes_from(nodes)
    dag.add_edges_from([(nodes[i], nodes[i+1]) for i in range(num_nodes - 1)])
    return dag

def get_fork_dag() -> nx.DiGraph:
    """Returns a fork DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C", "D", "E", "F"])
    dag.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "E"), ("D", "F"), ("E", "F")])
    return dag

def get_branching_dag(levels: int = 3, branching_factor: int = 2) -> nx.DiGraph:
    """Returns a branching DAG.

    Args:
        levels (int, optional): The number of levels. Defaults to 3.
        branching_factor (int, optional): The branching factor. Defaults to 2.

    Returns:
        nx.DiGraph: The branching DAG.
    """
    graph = nx.DiGraph()

    node_id = 0
    level_nodes = [node_id]  # Level 0
    graph.add_node(node_id)
    node_id += 1

    for _ in range(1, levels):
        new_level_nodes = []

        for parent in level_nodes:
            children = [node_id + i for i in range(branching_factor)]

            graph.add_edges_from([(parent, child) for child in children])
            new_level_nodes.extend(children)
            node_id += branching_factor

        level_nodes = new_level_nodes

    # Add destination node
    dst_node = node_id
    graph.add_node(dst_node)
    graph.add_edges_from([(node, dst_node) for node in level_nodes])

    return graph

def get_network(num_nodes = 4) -> nx.Graph:
    """Returns a network."""
    network = nx.Graph()
    network.add_nodes_from(range(num_nodes))
    # fully connected
    network.add_edges_from(product(range(num_nodes), range(num_nodes)))
    return network

# template T nx.DiGraph or nx.Graph
T = TypeVar("T", nx.DiGraph, nx.Graph)
def add_random_weights(graph: T, weight_range: Tuple[float, float] = (0, 1)) -> T:
    """Adds random weights to the DAG."""
    for node in graph.nodes:
        graph.nodes[node]["weight"] = np.random.uniform(*weight_range)
    for edge in graph.edges:
        if not graph.is_directed() and edge[0] == edge[1]:
            graph.edges[edge]["weight"] = 1e9 * weight_range[1] # very large communication speed
        else:
            graph.edges[edge]["weight"] = np.random.uniform(*weight_range)
    return graph

def add_rv_weights(graph: T, num_samples: int = 100) -> T:
    """Adds random variable weights to the DAG."""
    def get_rv():
        std = np.random.uniform(1e-9, 0.01)
        loc = np.random.uniform(0.5)
        x_vals = np.linspace(1e-9, 1, num_samples)
        pdf = norm.pdf(x_vals, loc, std)
        return RandomVariable.from_pdf(x_vals, pdf)

    for node in graph.nodes:
        graph.nodes[node]["weight"] = get_rv()
    for edge in graph.edges:
        if not graph.is_directed() and edge[0] == edge[1]:
            graph.edges[edge]["weight"] = RandomVariable([1e9]) # very large communication speed
        else:
            graph.edges[edge]["weight"] = get_rv()
    return graph
