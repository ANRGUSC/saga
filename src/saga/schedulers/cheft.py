"""Conditional HEFT (CHEFT) scheduler.

Pre-weights task costs and edge sizes by compound probability, then runs
standard HEFT.  Tasks on high-probability paths get higher upward rank and
therefore better node placements.

When the task graph has no conditional edges, all probabilities are 1.0 and
CHEFT degrades to standard HEFT.
"""

from typing import Dict, List, Optional

import numpy as np

from saga import Network, Schedule, ScheduledTask, Scheduler, TaskGraph, TaskGraphNode, TaskGraphEdge
from saga.schedulers.cpop import upward_rank
from saga.utils.duplication import should_duplicate


def compound_probabilities(task_graph: TaskGraph) -> Dict[str, float]:
    """Probability that each task executes (sum of trace probs it appears in).

    Returns 1.0 for every task when the graph has no conditional edges.
    """
    from saga.conditional import ConditionalTaskGraph

    tasks = [t.name for t in task_graph.tasks]

    if not isinstance(task_graph, ConditionalTaskGraph):
        return {t: 1.0 for t in tasks}

    traces = task_graph.identify_traces_detailed()
    probs: Dict[str, float] = {t: 0.0 for t in tasks}
    for trace in traces:
        for task_name in trace["tasks"]:
            probs[task_name] += trace["probability"]
    return probs


def build_weighted_graph(task_graph: TaskGraph) -> TaskGraph:
    """Return a plain TaskGraph with costs and edge sizes scaled by
    compound probability.

    - ``task.cost *= compound_prob(task)``
    - ``edge.size *= compound_prob(edge.target)``
    """
    probs = compound_probabilities(task_graph)

    weighted_tasks = frozenset(
        TaskGraphNode(name=t.name, cost=t.cost * probs[t.name])
        for t in task_graph.tasks
    )
    weighted_edges = frozenset(
        TaskGraphEdge(source=e.source, target=e.target, size=e.size * probs[e.target])
        for e in task_graph.dependencies
    )
    return TaskGraph(tasks=weighted_tasks, dependencies=weighted_edges)


def cheft_rank_sort(network: Network, task_graph: TaskGraph) -> List[str]:
    """Sort tasks by upward rank of the probability-weighted graph."""
    weighted = build_weighted_graph(task_graph)
    urank = upward_rank(network, weighted)
    topological_sort = {
        node.name: i for i, node in enumerate(reversed(task_graph.topological_sort()))
    }
    rank = {node: (urank[node], topological_sort[node]) for node in urank}
    return sorted(rank.keys(), key=lambda x: rank.get(x, 0.0), reverse=True)


class CheftScheduler(Scheduler):
    """HEFT with probability-weighted upward rank for conditional task graphs."""

    duplication_factor: int = 1

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        schedule_order = cheft_rank_sort(network, task_graph)
        schedule = Schedule(task_graph, network)

        for task_name in schedule_order:
            if schedule.is_scheduled(task_name):
                continue

            duplicate_factor = 1
            if should_duplicate(task_name, task_graph, network):
                duplicate_factor = self.duplication_factor

            for dup_idx in range(duplicate_factor):
                best_node = None
                best_finish_time = np.inf

                for node in network.nodes:
                    start_time = schedule.get_earliest_start_time(
                        task=task_name, node=node, append_only=False
                    )
                    start_time = max(start_time, min_start_time)
                    runtime = (
                        task_graph.get_task(task_name).cost / network.get_node(node).speed
                    )
                    finish_time = start_time + runtime
                    if finish_time < best_finish_time:
                        best_finish_time = finish_time
                        best_node = node

                if best_node is None:
                    raise ValueError(
                        f"No node available to schedule task {task_name}."
                    )

                new_task = ScheduledTask(
                    node=best_node.name,
                    name=task_name,
                    start=best_finish_time
                    - (task_graph.get_task(task_name).cost / best_node.speed),
                    end=best_finish_time,
                )
                schedule.add_task(new_task)

        return schedule
