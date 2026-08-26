"""Candidate scoring heuristics for choosing which tasks to duplicate.

Each score maps a task to a number in [0, 1]; higher means "more likely to be a
good duplication target". ``task_score`` averages them. These are experimental:
they are evaluated by ``run.py`` and are not wired into any SAGA scheduler.
"""

import networkx as nx

from saga import TaskGraph


def is_super_node(task_name: str) -> bool:
    """Whether a task is a synthetic super source/sink added by TaskGraph.create."""
    return task_name.startswith("__super_")


def task_can_be_scored(task_name: str, task_graph: TaskGraph) -> bool:
    """A task can be scored only if it is real and has at least one real child."""
    if is_super_node(task_name):
        return False
    real_children = [
        edge.target
        for edge in task_graph.out_edges(task_name)
        if not is_super_node(edge.target)
    ]
    return len(real_children) > 0


def communication_score(task_name: str, task_graph: TaskGraph) -> float:
    """Outgoing vs incoming data: high when a task produces far more than it consumes."""
    if not task_can_be_scored(task_name, task_graph):
        return 0.0

    outgoing_sum = sum(
        edge.size
        for edge in task_graph.out_edges(task_name)
        if not is_super_node(edge.target)
    )
    incoming_sum = sum(
        edge.size
        for edge in task_graph.in_edges(task_name)
        if not is_super_node(edge.source)
    )
    if outgoing_sum + incoming_sum == 0:
        return 0.0
    return max(0.0, (outgoing_sum - incoming_sum) / (outgoing_sum + incoming_sum))


def impact_score(task_name: str, task_graph: TaskGraph) -> float:
    """Fraction of total computation that descends from this task."""
    if is_super_node(task_name) or not task_can_be_scored(task_name, task_graph):
        return 0.0

    descendants = {
        descendant
        for descendant in nx.descendants(task_graph.graph, task_name)
        if not is_super_node(descendant)
    }
    descendant_compute = sum(
        task_graph.get_task(descendant).cost for descendant in descendants
    )
    total_graph_compute = sum(
        task.cost for task in task_graph.tasks if not is_super_node(task.name)
    )
    if total_graph_compute == 0:
        return 0.0
    return descendant_compute / total_graph_compute


def branching_score(task_name: str, task_graph: TaskGraph) -> float:
    """Number of children relative to the most-branching task in the graph."""
    if is_super_node(task_name) or not task_can_be_scored(task_name, task_graph):
        return 0.0

    task_out_degree = sum(
        1 for edge in task_graph.out_edges(task_name) if not is_super_node(edge.target)
    )
    max_out_degree = max(
        sum(
            1
            for edge in task_graph.out_edges(task.name)
            if not is_super_node(edge.target)
        )
        for task in task_graph.tasks
        if not is_super_node(task.name)
    )
    if max_out_degree == 0:
        return 0.0
    return task_out_degree / max_out_degree


def join_val_score(task_name: str, task_graph: TaskGraph) -> float:
    """How much this task dominates the inputs of any child that joins several parents."""
    if not task_can_be_scored(task_name, task_graph):
        return 0.0

    scores = []
    for child_edge in task_graph.out_edges(task_name):
        child_name = child_edge.target
        if task_graph.in_degree(child_name) < 2:
            continue
        max_incoming = max(edge.size for edge in task_graph.in_edges(child_name))
        if max_incoming == 0:
            continue
        scores.append(child_edge.size / max_incoming)
    if not scores:
        return 0.0
    return max(scores)


def task_score(task_name: str, task_graph: TaskGraph) -> float:
    """Combined heuristic: the mean of the four component scores."""
    return (
        communication_score(task_name, task_graph)
        + impact_score(task_name, task_graph)
        + branching_score(task_name, task_graph)
        + join_val_score(task_name, task_graph)
    ) / 4
