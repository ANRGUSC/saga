from itertools import product
from typing import Callable, Optional, TypeVar
import networkx as nx
import numpy as np
from saga import Network, TaskGraph
from scipy.stats import norm

from saga.utils.random_variable import RandomVariable, UniformRandomVariable


DEFAULT_WEIGHT_DISTRIBUTION = UniformRandomVariable(0.1, 1.0)

T = TypeVar("T", nx.DiGraph, nx.Graph)


def add_random_weights(
    graph: T,
    weight_distribution: Optional[RandomVariable] = None,
    node_weight_distribution: Optional[RandomVariable] = None,
    edge_weight_distribution: Optional[RandomVariable] = None,
) -> T:
    """Adds random weights to the graph.

    Args:
        graph: The graph to add weights to.
        weight_distribution: Distribution for both nodes and edges (default).
        node_weight_distribution: Distribution for node weights (overrides weight_distribution).
        edge_weight_distribution: Distribution for edge weights (overrides weight_distribution).

    Returns:
        The graph with weights added.
    """
    if weight_distribution is None:
        weight_distribution = DEFAULT_WEIGHT_DISTRIBUTION

    node_dist = node_weight_distribution or weight_distribution
    edge_dist = edge_weight_distribution or weight_distribution

    for node in graph.nodes:
        graph.nodes[node]["weight"] = node_dist.sample()
    for edge in graph.edges:
        if not graph.is_directed() and edge[0] == edge[1]:
            graph.edges[edge]["weight"] = (
                1e9 * edge_dist.sample()
            )  # very large communication speed
        else:
            graph.edges[edge]["weight"] = edge_dist.sample()
    return graph


def default_get_rv(num_samples: int = 100) -> RandomVariable:
    std = np.random.uniform(1e-9, 0.01)
    loc = np.random.uniform(0.5)
    x_vals = np.linspace(1e-9, 1, num_samples)
    pdf = norm.pdf(x_vals, loc, std)
    return RandomVariable.from_pdf(x_vals, pdf)


def add_rv_weights(
    graph: T, get_rv: Callable[[], RandomVariable] = default_get_rv
) -> T:
    """Adds random variable weights to the DAG."""
    for node in graph.nodes:
        graph.nodes[node]["weight"] = get_rv()
    for edge in graph.edges:
        if not graph.is_directed() and edge[0] == edge[1]:
            graph.edges[edge]["weight"] = RandomVariable(samples=[1e9])
        else:
            graph.edges[edge]["weight"] = get_rv()
    return graph


def get_diamond_dag(
    weight_distribution: Optional[RandomVariable] = None,
    node_weight_distribution: Optional[RandomVariable] = None,
    edge_weight_distribution: Optional[RandomVariable] = None,
) -> TaskGraph:
    """Returns a diamond DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C", "D"])
    dag.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    dag = add_random_weights(
        dag, weight_distribution, node_weight_distribution, edge_weight_distribution
    )
    return TaskGraph.from_nx(dag)


def get_chain_dag(
    num_nodes: int = 4,
    weight_distribution: Optional[RandomVariable] = None,
    node_weight_distribution: Optional[RandomVariable] = None,
    edge_weight_distribution: Optional[RandomVariable] = None,
) -> TaskGraph:
    """Returns a chain DAG."""
    dag = nx.DiGraph()
    nodes = [chr(ord("A") + i) for i in range(num_nodes)]
    dag.add_nodes_from(nodes)
    dag.add_edges_from([(nodes[i], nodes[i + 1]) for i in range(num_nodes - 1)])
    dag = add_random_weights(
        dag, weight_distribution, node_weight_distribution, edge_weight_distribution
    )
    return TaskGraph.from_nx(dag)


def get_fork_dag(
    weight_distribution: Optional[RandomVariable] = None,
    node_weight_distribution: Optional[RandomVariable] = None,
    edge_weight_distribution: Optional[RandomVariable] = None,
) -> TaskGraph:
    """Returns a fork DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C", "D", "E", "F"])
    dag.add_edges_from(
        [("A", "B"), ("A", "C"), ("B", "D"), ("C", "E"), ("D", "F"), ("E", "F")]
    )
    dag = add_random_weights(
        dag, weight_distribution, node_weight_distribution, edge_weight_distribution
    )
    return TaskGraph.from_nx(dag)


def get_branching_dag(
    levels: int = 3,
    branching_factor: int = 2,
    weight_distribution: Optional[RandomVariable] = None,
    node_weight_distribution: Optional[RandomVariable] = None,
    edge_weight_distribution: Optional[RandomVariable] = None,
) -> TaskGraph:
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
    graph.add_node(str(node_id))
    node_id += 1

    for _ in range(1, levels):
        new_level_nodes = []

        for parent in level_nodes:
            children = [node_id + i for i in range(branching_factor)]

            graph.add_edges_from([(str(parent), str(child)) for child in children])
            new_level_nodes.extend(children)
            node_id += branching_factor

        level_nodes = new_level_nodes

    # Add destination node
    dst_node = node_id
    graph.add_node(str(dst_node))
    graph.add_edges_from([(str(node), str(dst_node)) for node in level_nodes])

    graph = add_random_weights(
        graph, weight_distribution, node_weight_distribution, edge_weight_distribution
    )
    return TaskGraph.from_nx(graph)


def get_network(
    num_nodes: int = 4,
    weight_distribution: Optional[RandomVariable] = None,
    node_weight_distribution: Optional[RandomVariable] = None,
    edge_weight_distribution: Optional[RandomVariable] = None,
) -> Network:
    """Returns a fully-connected network.

    Args:
        num_nodes: Number of nodes in the network.
        weight_distribution: Distribution for both nodes and edges (default).
        node_weight_distribution: Distribution for node weights (computation speed).
        edge_weight_distribution: Distribution for edge weights (communication speed).

    Returns:
        A fully-connected Network.
    """
    network = nx.Graph()
    node_names = list(map(str, range(num_nodes)))
    network.add_nodes_from(node_names)
    network.add_edges_from(product(node_names, node_names))
    network = add_random_weights(
        network, weight_distribution, node_weight_distribution, edge_weight_distribution
    )
    return Network.from_nx(network)
