from itertools import product
from typing import Callable, Optional, TypeVar
import networkx as nx
import numpy as np
from saga import Network, TaskGraph, TaskGraphNode
from saga.conditional import ConditionalTaskGraph, ConditionalTaskGraphEdge
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
        graph.nodes[node]["weight"] = float(node_dist.sample()[0])
    for edge in graph.edges:
        if not graph.is_directed() and edge[0] == edge[1]:
            graph.edges[edge]["weight"] = 1e9 * float(
                edge_dist.sample()[0]
            )  # very large communication speed
        else:
            graph.edges[edge]["weight"] = float(edge_dist.sample()[0])
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


def _from_nx_conditional(dag: nx.DiGraph) -> ConditionalTaskGraph:
    tasks = [
        TaskGraphNode(name=str(node), cost=float(dag.nodes[node]["weight"]))
        for node in dag.nodes
    ]
    dependencies = [
        ConditionalTaskGraphEdge(
            source=str(u),
            target=str(v),
            size=float(dag.edges[u, v]["weight"]),
            probability=float(dag.edges[u, v].get("probability", 1.0)),
        )
        for u, v in dag.edges
    ]
    return ConditionalTaskGraph.create(tasks=tasks, dependencies=dependencies)


def add_conditional_probabilities(
    dag: nx.DiGraph,
    conditional_parent_probability: float = 0.4,
) -> nx.DiGraph:
    """Adds conditional probabilities to fan-out edges.

    For each node with more than one outgoing edge:
    - with probability `conditional_parent_probability`, outgoing edges are
      treated as conditional alternatives and assigned probabilities summing to 1.
    - otherwise, each outgoing edge gets probability 1.0.
    """
    for parent in dag.nodes:
        children = list(dag.successors(parent))
        if len(children) <= 1:
            continue

        if np.random.random() > conditional_parent_probability:
            for child in children:
                dag.edges[parent, child]["probability"] = 1.0
            continue

        branch_probs = np.random.random(len(children))
        branch_probs = branch_probs / branch_probs.sum()
        for child, prob in zip(children, branch_probs):
            dag.edges[parent, child]["probability"] = float(prob)

    return dag


def get_conditional_diamond_dag(
    weight_distribution: Optional[RandomVariable] = None,
    node_weight_distribution: Optional[RandomVariable] = None,
    edge_weight_distribution: Optional[RandomVariable] = None,
    branch_probability: float = 0.5,
) -> ConditionalTaskGraph:
    """Returns a conditional diamond DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C", "D"])
    dag.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    dag = add_random_weights(
        dag, weight_distribution, node_weight_distribution, edge_weight_distribution
    )
    dag.edges["A", "B"]["probability"] = branch_probability
    dag.edges["A", "C"]["probability"] = 1.0 - branch_probability
    dag.edges["B", "D"]["probability"] = 1.0
    dag.edges["C", "D"]["probability"] = 1.0
    return _from_nx_conditional(dag)


def get_random_conditional_branching_dag(
    levels: int = 3,
    branching_factor: int = 2,
    conditional_parent_probability: float = 0.4,
    weight_distribution: Optional[RandomVariable] = None,
    node_weight_distribution: Optional[RandomVariable] = None,
    edge_weight_distribution: Optional[RandomVariable] = None,
) -> ConditionalTaskGraph:
    """Returns a branching DAG with random conditional branches."""
    dag = nx.DiGraph()

    node_id = 0
    level_nodes = [node_id]
    dag.add_node(str(node_id))
    node_id += 1

    for _ in range(1, levels):
        new_level_nodes = []
        for parent in level_nodes:
            children = [node_id + i for i in range(branching_factor)]
            dag.add_edges_from([(str(parent), str(child)) for child in children])
            new_level_nodes.extend(children)
            node_id += branching_factor
        level_nodes = new_level_nodes

    sink = node_id
    dag.add_node(str(sink))
    dag.add_edges_from([(str(node), str(sink)) for node in level_nodes])

    dag = add_random_weights(
        dag, weight_distribution, node_weight_distribution, edge_weight_distribution
    )
    dag = add_conditional_probabilities(
        dag, conditional_parent_probability=conditional_parent_probability
    )
    return _from_nx_conditional(dag)

