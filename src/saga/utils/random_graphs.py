from itertools import product
from typing import Callable, Dict, Optional, Tuple, TypeVar, cast
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
    # ``create`` is a classmethod that builds ``cls``; its declared return type
    # is the base ``TaskGraph``, so narrow it back to the concrete subclass.
    return cast(
        ConditionalTaskGraph,
        ConditionalTaskGraph.create(tasks=tasks, dependencies=dependencies),
    )


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

        alternative_probs = np.random.random(len(children))
        alternative_probs = alternative_probs / alternative_probs.sum()
        for child, prob in zip(children, alternative_probs):
            dag.edges[parent, child]["probability"] = float(prob)

    return dag


def get_conditional_diamond_dag(
    weight_distribution: Optional[RandomVariable] = None,
    node_weight_distribution: Optional[RandomVariable] = None,
    edge_weight_distribution: Optional[RandomVariable] = None,
    trace_probability: float = 0.5,
) -> ConditionalTaskGraph:
    """Returns a conditional diamond DAG.

    Args:
        trace_probability: Probability of the trace that takes edge A -> B.
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C", "D"])
    dag.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    dag = add_random_weights(
        dag, weight_distribution, node_weight_distribution, edge_weight_distribution
    )
    dag.edges["A", "B"]["probability"] = trace_probability
    dag.edges["A", "C"]["probability"] = 1.0 - trace_probability
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
    """Returns a branching DAG with random conditional alternatives."""
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


def get_conditional_dag(
    max_depth: int = 3,
    branch_factor: int = 2,
    leaf_probability: float = 0.25,
    conditional_probability: float = 0.5,
    series_probability: float = 0.3,
    branch_alpha: float = 1.0,
    weight_distribution: Optional[RandomVariable] = None,
    node_weight_distribution: Optional[RandomVariable] = None,
    edge_weight_distribution: Optional[RandomVariable] = None,
) -> ConditionalTaskGraph:
    """Generate a nested series/parallel/conditional DAG (the real-time CTG model).

    Follows the real-time-systems "conditional DAG task" structure (Melani et al.,
    ECRTS 2015) with per-branch probabilities as in its probabilistic variant
    (Ueter, Guenzel, Chen, 2021): the graph is built recursively as a
    series-parallel composition where each composite block is either

    - a **series** chain of sub-blocks,
    - a **parallel** (AND) fork/join where every branch runs, or
    - a **conditional** (XOR) fork/join where exactly one branch runs, with the
      branch chosen by a probability drawn from a Dirichlet(``branch_alpha``).

    Nested conditionals reconverge at their join node, so a single graph unfolds
    into many mutually exclusive traces (exactly one branch per conditional fork).
    The fork node of a conditional block plays the role of the "decision" task.

    Args:
        max_depth: Maximum recursion depth (controls size).
        branch_factor: Maximum children per composite block (2..branch_factor).
        leaf_probability: Probability of stopping recursion at a single task.
        conditional_probability: Among fork/join blocks, the fraction made
            conditional (XOR) rather than parallel (AND).
        series_probability: Probability a composite block is a series chain.
        branch_alpha: Dirichlet concentration for conditional branch probabilities.
            Small values (< 1) produce skewed branches (one likely, others rare);
            large values produce near-uniform branches.
        weight_distribution: Default distribution for node and edge weights.
        node_weight_distribution: Overrides ``weight_distribution`` for task costs.
        edge_weight_distribution: Overrides ``weight_distribution`` for data sizes.

    Returns:
        A :class:`~saga.conditional.ConditionalTaskGraph` with one source and one
        sink.
    """
    weight_distribution = weight_distribution or DEFAULT_WEIGHT_DISTRIBUTION
    node_dist = node_weight_distribution or weight_distribution
    edge_dist = edge_weight_distribution or weight_distribution

    tasks: Dict[str, float] = {}
    edges: list = []  # (source, target, size, probability)
    counter = [0]

    def new_task() -> str:
        name = str(counter[0])
        counter[0] += 1
        tasks[name] = float(node_dist.sample()[0])
        return name

    def add_edge(source: str, target: str, probability: float = 1.0) -> None:
        edges.append((source, target, float(edge_dist.sample()[0]), probability))

    def build(depth: int) -> Tuple[str, str]:
        """Build one block; return its (entry, exit) task names."""
        if depth <= 0 or np.random.random() < leaf_probability:
            task = new_task()
            return task, task

        num_children = int(np.random.randint(2, branch_factor + 1))
        if np.random.random() < series_probability:
            blocks = [build(depth - 1) for _ in range(num_children)]
            for (_, prev_exit), (next_entry, _) in zip(blocks, blocks[1:]):
                add_edge(prev_exit, next_entry)
            return blocks[0][0], blocks[-1][1]

        fork, join = new_task(), new_task()
        children = [build(depth - 1) for _ in range(num_children)]
        if np.random.random() < conditional_probability:
            probs = np.random.dirichlet([branch_alpha] * num_children)
            for (child_entry, child_exit), prob in zip(children, probs):
                add_edge(fork, child_entry, probability=float(prob))
                add_edge(child_exit, join)
        else:
            for child_entry, child_exit in children:
                add_edge(fork, child_entry)
                add_edge(child_exit, join)
        return fork, join

    build(max_depth)
    return cast(
        ConditionalTaskGraph,
        ConditionalTaskGraph.create(
            tasks=[(name, cost) for name, cost in tasks.items()],
            dependencies=[
                ConditionalTaskGraphEdge(source=s, target=t, size=sz, probability=p)
                for s, t, sz, p in edges
            ],
        ),
    )


def get_problematic_instance(
    likely_probability: Optional[float] = None,
    shared_cost: Optional[float] = None,
    unlikely_cost_factor: Optional[float] = None,
    light_cost: Optional[float] = None,
    fast_speed: Optional[float] = None,
    slow_speed: Optional[float] = None,
) -> Tuple[Network, ConditionalTaskGraph]:
    """Return a random instance from a family where probability reweighting wins big.

    This is an adversarial family for the *unweighted* conditional scheduler (plain
    HEFT on a CTG): a scheduler that reweights task priority by execution probability
    (CHEFT) reliably achieves much lower expected makespan.

    Structure (``R`` is the entry, ``Z`` the sink)::

        R -> P            (shared: non-conditional, runs in every trace)
        R -> H            (conditional, UNLIKELY branch, probability 1 - p)
        R -> L            (conditional, LIKELY branch, probability p)
        P, H, L -> Z

    ``P`` is a heavy shared task; ``H`` is an even heavier task on the unlikely
    branch; ``L`` is light. The network has one fast node and one slow node.

    Why the unweighted scheduler loses: because a trace's realized makespan depends
    only on how its tasks are *assigned* to nodes, and ``P`` co-occurs with ``H`` in
    the unlikely trace (so they are not mutually exclusive and cannot share a node),
    the unweighted upward rank puts the heavy ``H`` first and lets it take the fast
    node, pushing the shared ``P`` onto the slow node. ``P`` also runs in the likely
    trace, so the (probability-``p``) likely trace pays for ``P`` on the slow node.
    Reweighting scales ``H``'s priority by its small probability ``1 - p``, so ``P``
    takes the fast node and the likely trace is fast. The advantage grows with ``p``,
    and with the fast/slow speed ratio up to the point where ``H`` stops being heavy
    enough to displace ``P`` (see ``unlikely_cost_factor``), beyond which it falls off.

    Args:
        likely_probability: Probability ``p`` of the likely branch. Sampled from
            ``[0.88, 0.97]`` if None.
        shared_cost: Cost of the shared task ``P``. Sampled from ``[8, 12]`` if None.
        unlikely_cost_factor: ``H``'s cost as a multiple of ``P``'s. For the effect
            it must exceed ``fast_speed / slow_speed - 1`` (so the unweighted EFT
            prefers to put ``P`` on the slow node rather than queue it behind ``H``
            on the fast node); otherwise ``P`` stays fast and the advantage vanishes.
            Sampled from ``[4, 6]`` if None.
        light_cost: Cost of the light task ``L``. Sampled from ``[0.5, 2]`` if None.
        fast_speed: Speed of the fast node. Sampled from ``[1.9, 2.1]`` if None.
        slow_speed: Speed of the slow node. Sampled from ``[0.4, 0.55]`` if None.

    Returns:
        A ``(network, task_graph)`` pair. The task graph is a
        :class:`~saga.conditional.ConditionalTaskGraph` with two traces:
        the likely ``{R, P, L, Z}`` (probability ``p``) and the unlikely
        ``{R, P, H, Z}`` (probability ``1 - p``).
    """
    rng = np.random
    p = likely_probability if likely_probability is not None else float(rng.uniform(0.88, 0.97))
    p_cost = shared_cost if shared_cost is not None else float(rng.uniform(8.0, 12.0))
    factor = (
        unlikely_cost_factor
        if unlikely_cost_factor is not None
        else float(rng.uniform(4.0, 6.0))
    )
    l_cost = light_cost if light_cost is not None else float(rng.uniform(0.5, 2.0))
    fast = fast_speed if fast_speed is not None else float(rng.uniform(1.9, 2.1))
    slow = slow_speed if slow_speed is not None else float(rng.uniform(0.4, 0.55))
    r_cost, z_cost = float(rng.uniform(0.2, 1.0)), float(rng.uniform(0.2, 1.0))

    def size() -> float:
        return float(rng.uniform(0.1, 1.0))

    task_graph = ConditionalTaskGraph.create(
        tasks=[
            ("R", r_cost),
            ("P", p_cost),
            ("H", factor * p_cost),
            ("L", l_cost),
            ("Z", z_cost),
        ],
        dependencies=[
            ConditionalTaskGraphEdge(source="R", target="P", size=size()),
            ConditionalTaskGraphEdge(source="R", target="H", size=size(), probability=1.0 - p),
            ConditionalTaskGraphEdge(source="R", target="L", size=size(), probability=p),
            ConditionalTaskGraphEdge(source="P", target="Z", size=size()),
            ConditionalTaskGraphEdge(source="H", target="Z", size=size()),
            ConditionalTaskGraphEdge(source="L", target="Z", size=size()),
        ],
    )
    network = Network.create(
        nodes=[("fast", fast), ("slow", slow)],
        edges=[("fast", "slow", float(rng.uniform(1.0, 5.0)))],
    )
    return network, cast(ConditionalTaskGraph, task_graph)

