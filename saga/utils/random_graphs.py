from itertools import product
from typing import Tuple, TypeVar
import networkx as nx
import numpy as np
from scipy.stats import norm

from saga.utils.random_variable import RandomVariable

def get_diamond_dag() -> nx.DiGraph:
    """Returns a diamond DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C", "D"])
    dag.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    return dag

def get_chain_dag() -> nx.DiGraph:
    """Returns a chain DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C", "D"])
    dag.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    return dag

def get_fork_dag() -> nx.DiGraph:
    """Returns a fork DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C", "D", "E", "F"])
    dag.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "E"), ("D", "F"), ("E", "F")])
    return dag

def get_branching_dag(levels: int = 3, branching_factor: int = 2) -> nx.DiGraph:
    G = nx.DiGraph()
    
    node_id = 0
    level_nodes = [node_id]  # Level 0
    G.add_node(node_id)
    node_id += 1
    
    for level in range(1, levels):
        new_level_nodes = []
        
        for parent in level_nodes:
            children = [node_id + i for i in range(branching_factor)]
            
            G.add_edges_from([(parent, child) for child in children])
            new_level_nodes.extend(children)
            node_id += branching_factor
        
        level_nodes = new_level_nodes
    
    # Add destination node
    dst_node = node_id
    G.add_node(dst_node)
    G.add_edges_from([(node, dst_node) for node in level_nodes])

    return G

def get_network() -> nx.Graph:
    """Returns a network."""
    network = nx.Graph()
    network.add_nodes_from(range(4))
    # fully connected
    network.add_edges_from(product(range(4), range(4)))
    return network

# template T nx.DiGraph or nx.Graph
T = TypeVar("T", nx.DiGraph, nx.Graph)
def add_random_weights(graph: T, weight_range: Tuple[float, float] = (1, 10)) -> T:
    """Adds random weights to the DAG."""
    for node in graph.nodes:
        graph.nodes[node]["weight"] = np.random.uniform(*weight_range)
    for edge in graph.edges:
        if not graph.is_directed() and edge[0] == edge[1]:
            graph.edges[edge]["weight"] = 1e9 * weight_range[1] # very large communication speed
        else:
            graph.edges[edge]["weight"] = np.random.uniform(*weight_range)
    return graph

def add_rv_weights(graph: T) -> T:
    """Adds random variable weights to the DAG."""
    def get_rv():
        std = np.random.uniform(1e-9, 0.01)
        loc = np.random.uniform(0.5)
        x = np.linspace(1e-9, 1, 1000)
        pdf = norm.pdf(x, loc, std)
        return RandomVariable.from_pdf(x, pdf)
    
    for node in graph.nodes:
        graph.nodes[node]["weight"] = get_rv()
    for edge in graph.edges:
        if not graph.is_directed() and edge[0] == edge[1]:
            graph.edges[edge]["weight"] = RandomVariable([1e9]) # very large communication speed
        else:
            graph.edges[edge]["weight"] = get_rv()
    return graph
