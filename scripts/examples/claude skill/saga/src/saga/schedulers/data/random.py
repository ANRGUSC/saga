import random
from typing import Callable, Hashable, List, Optional
import networkx as nx

from saga import TaskGraph, Network


# Task Graph Datasets
def _default_task_weight(task: Hashable) -> float:
    """Default task weight function.

    Args:
        task (Hashable): The task.

    Returns:
        float: The task weight.
    """
    return max(min(1e-9, random.gauss(1, 1 / 3)), 2)


def _default_dependency_weight(src: Hashable, dst: Hashable) -> float:
    """Default dependency weight function.

    Args:
        src (Hashable): The source node.
        dst (Hashable): The destination node.

    Returns:
        float: The dependency weight.
    """
    return max(min(1e-9, random.gauss(1, 1 / 3)), 2)


def gen_out_trees(
    num: int,
    num_levels: int,
    branching_factor: int,
    get_task_weight: Optional[Callable[[Hashable], float]] = None,
    get_dependency_weight: Optional[Callable[[Hashable, Hashable], float]] = None,
) -> List[TaskGraph]:
    """Generate a dataset of in-trees.

    Args:
        num: Number of graphs to generate.
        num_levels: Number of levels in the tree.
        branching_factor: Number of parents per node.
        get_task_weight: A function that returns the weight of a task.
        get_dependency_weight: A function that returns the weight of a dependency.

    Returns:
        A list of out-trees.
    """
    assert num > 0
    assert num_levels > 0
    assert branching_factor > 0

    if get_task_weight is None:
        get_task_weight = _default_task_weight

    if get_dependency_weight is None:
        get_dependency_weight = _default_dependency_weight

    trees: List[nx.DiGraph] = []
    for _ in range(num):
        tree: nx.DiGraph = nx.generators.balanced_tree(branching_factor, num_levels)
        tree = nx.DiGraph(tree)
        tree.remove_edges_from([(dst, src) for src, dst in tree.edges if src < dst])
        tree = nx.relabel_nodes(tree, mapping={node: f"T{node}" for node in tree.nodes})
        for node in tree.nodes:
            tree.nodes[node]["weight"] = get_task_weight(node)
        for edge in tree.edges:
            tree.edges[edge]["weight"] = get_dependency_weight(edge[0], edge[1])
        # add sink node
        sink_node = f"T{len(tree.nodes)}"
        tree.add_node(sink_node, weight=1e-9)
        leaf_nodes = [
            node
            for node in tree.nodes
            if tree.out_degree(node) == 0 and node != sink_node
        ]
        for node in leaf_nodes:
            tree.add_edge(node, sink_node, weight=1e-9)
        trees.append(tree)

    return [TaskGraph.from_nx(tree) for tree in trees]


def gen_in_trees(
    num: int,  # pylint: disable=arguments-differ
    num_levels: int,
    branching_factor: int,
    get_task_weight: Optional[Callable[[Hashable], float]] = None,
    get_dependency_weight: Optional[Callable[[Hashable, Hashable], float]] = None,
) -> List[TaskGraph]:
    """Generate a dataset of in-trees.

    Args:
        num: Number of graphs to generate.
        num_levels: Number of levels in the tree.
        branching_factor: Number of parents per node.
        get_task_weight: A function that returns the weight of a task.
        get_dependency_weight: A function that returns the weight of a dependency.

    Returns:
        A list of in-trees.
    """
    if get_task_weight is None:
        get_task_weight = _default_task_weight

    if get_dependency_weight is None:
        get_dependency_weight = _default_dependency_weight

    out_trees = gen_out_trees(
        num, num_levels, branching_factor, get_task_weight, get_dependency_weight
    )
    in_trees = []
    for tree in out_trees:
        in_trees.append(tree.graph.reverse())

    return [TaskGraph.from_nx(tree) for tree in in_trees]


def gen_parallel_chains(
    num: int,
    num_chains: int,
    chain_length: int,
    get_task_weight: Optional[Callable[[Hashable], float]] = None,
    get_dependency_weight: Optional[Callable[[Hashable, Hashable], float]] = None,
) -> List[TaskGraph]:
    """Generate a dataset of parallel chains.

    Args:
        num: Number of graphs to generate.
        num_chains: Number of chains in the graph.
        chain_length: Length of each chain.
        get_task_weight: A function that returns the weight of a task.
        get_dependency_weight: A function that returns the weight of a dependency.

    Returns:
        A dataset of parallel chains.
    """
    assert num > 0
    assert num_chains > 0
    assert chain_length > 0

    if get_task_weight is None:
        get_task_weight = _default_task_weight

    if get_dependency_weight is None:
        get_dependency_weight = _default_dependency_weight

    graphs: List[nx.DiGraph] = []
    for _ in range(num):
        graph = nx.DiGraph()
        graph.add_node("T0", weight=get_task_weight(0))
        node_count = 1
        endpoints = []
        for _ in range(num_chains):
            graph.add_node(f"T{node_count}", weight=get_task_weight(f"T{node_count}"))
            graph.add_edge(
                "T0",
                f"T{node_count}",
                weight=get_dependency_weight("T0", f"T{node_count}"),
            )
            node_count += 1
            for _ in range(1, chain_length):
                graph.add_node(
                    f"T{node_count}", weight=get_task_weight(f"T{node_count}")
                )
                graph.add_edge(
                    f"T{node_count - 1}",
                    f"T{node_count}",
                    weight=get_dependency_weight(
                        f"T{node_count - 1}", f"T{node_count}"
                    ),
                )
                node_count += 1
            endpoints.append(f"T{node_count - 1}")

        graph.add_node(f"T{node_count}", weight=get_task_weight(f"T{node_count}"))
        for endpoint in endpoints:
            graph.add_edge(
                endpoint,
                f"T{node_count}",
                weight=get_dependency_weight(endpoint, f"T{node_count}"),
            )

        graphs.append(graph)

    return [TaskGraph.from_nx(graph) for graph in graphs]


# Network Datasets
def gen_random_networks(
    num: int,
    num_nodes: int,
    get_node_weight: Optional[Callable[[Hashable], float]] = None,
    get_edge_weight: Optional[Callable[[Hashable, Hashable], float]] = None,
) -> List[Network]:
    """Generate a dataset of random networks.

    Args:
        num: Number of graphs to generate.
        num_nodes: Number of nodes in each graph.
        get_node_weight: A function that returns the weight of a node.
        get_edge_weight: A function that returns the weight of an edge.

    Returns:
        A dataset of random networks.
    """
    assert num > 0
    assert num_nodes > 0

    if get_node_weight is None:
        get_node_weight = _default_task_weight

    if get_edge_weight is None:
        get_edge_weight = _default_dependency_weight

    graphs: List[nx.Graph] = []
    for _ in range(num):
        graph = nx.generators.complete_graph(num_nodes)
        graph = nx.Graph(graph)
        graph = nx.relabel_nodes(
            graph, mapping={node: f"N{node}" for node in graph.nodes}
        )
        for edge in graph.edges:
            graph.edges[edge]["weight"] = get_edge_weight(edge[0], edge[1])
        for node in graph.nodes:
            graph.nodes[node]["weight"] = get_node_weight(node)
            graph.add_edge(node, node, weight=1e9)
        graphs.append(graph)

    return [Network.from_nx(graph) for graph in graphs]
