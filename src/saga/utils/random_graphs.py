from itertools import product
from typing import Callable, TypeVar
import networkx as nx
import numpy as np
from saga import Network, TaskGraph
from scipy.stats import norm

from saga.utils.random_variable import RandomVariable, UniformRandomVariable


DEFAULT_WEIGHT_DISTRIBUTION = UniformRandomVariable(0.1, 1.0)

T = TypeVar("T", nx.DiGraph, nx.Graph)
def add_random_weights(graph: T, weight_distribution: RandomVariable) -> T:
    """Adds random weights to the DAG."""
    for node in graph.nodes:
        graph.nodes[node]["weight"] = weight_distribution.sample()
    for edge in graph.edges:
        if not graph.is_directed() and edge[0] == edge[1]:
            graph.edges[edge]["weight"] = 1e9 * weight_distribution.sample() # very large communication speed
        else:
            graph.edges[edge]["weight"] = weight_distribution.sample()
    return graph

def default_get_rv(num_samples: int = 100) -> RandomVariable:
    std = np.random.uniform(1e-9, 0.01)
    loc = np.random.uniform(0.5)
    x_vals = np.linspace(1e-9, 1, num_samples)
    pdf = norm.pdf(x_vals, loc, std)
    return RandomVariable.from_pdf(x_vals, pdf)

def add_rv_weights(graph: T,
                   get_rv: Callable[[], RandomVariable] = default_get_rv) -> T:
    """Adds random variable weights to the DAG."""
    for node in graph.nodes:
        graph.nodes[node]["weight"] = get_rv()
    for edge in graph.edges:
        if not graph.is_directed() and edge[0] == edge[1]:
            graph.edges[edge]["weight"] = RandomVariable([1e9])
        else:
            graph.edges[edge]["weight"] = get_rv()
    return graph

def get_diamond_dag(weight_distribution: RandomVariable = DEFAULT_WEIGHT_DISTRIBUTION) -> TaskGraph:
    """Returns a diamond DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C", "D"])
    dag.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    dag = add_random_weights(dag, weight_distribution)
    return TaskGraph.from_nx(dag)

def get_chain_dag(num_nodes: int = 4, weight_distribution: RandomVariable = DEFAULT_WEIGHT_DISTRIBUTION) -> TaskGraph:
    """Returns a chain DAG."""
    dag = nx.DiGraph()
    nodes = [chr(ord("A") + i) for i in range(num_nodes)]
    dag.add_nodes_from(nodes)
    dag.add_edges_from([(nodes[i], nodes[i+1]) for i in range(num_nodes - 1)])
    dag = add_random_weights(dag, weight_distribution)
    return TaskGraph.from_nx(dag)

def get_fork_dag(weight_distribution: RandomVariable = DEFAULT_WEIGHT_DISTRIBUTION) -> TaskGraph:
    """Returns a fork DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C", "D", "E", "F"])
    dag.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "E"), ("D", "F"), ("E", "F")])
    dag = add_random_weights(dag, weight_distribution)
    return TaskGraph.from_nx(dag)

def get_branching_dag(levels: int = 3, branching_factor: int = 2, weight_distribution: RandomVariable = DEFAULT_WEIGHT_DISTRIBUTION) -> TaskGraph:
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

    graph = add_random_weights(graph, weight_distribution)
    return TaskGraph.from_nx(graph)

def get_network(num_nodes = 4, weight_distribution: RandomVariable = DEFAULT_WEIGHT_DISTRIBUTION) -> Network:
    """Returns a network."""
    network = nx.Graph()
    network.add_nodes_from(range(num_nodes))
    network.add_edges_from(product(range(num_nodes), range(num_nodes)))
    network = add_random_weights(network, weight_distribution)
    return Network.from_nx(network)
