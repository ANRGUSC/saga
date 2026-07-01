"""Helpers for scheduling conditional task graphs (CTGs).

When scheduling a CTG:
1) Identify all individual execution traces.
2) Build a mutual-exclusion graph linking tasks that can overlap.
3) During scheduling, use that graph to decide whether overlaps are allowed.
"""

from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from pydantic import Field, model_validator

from saga import (
    Network,
    Schedule,
    Scheduler,
    ScheduledTask,
    TaskGraph,
    TaskGraphEdge,
)


class ConditionalTaskGraphEdge(TaskGraphEdge):
    """Task graph edge with conditional probability metadata."""

    probability: float = Field(
        default=1.0,
        description=(
            "Conditional edge probability in [0, 1], assume conditionality "
            "if probability set to < 1."
        ),
    )

    @model_validator(mode="after")
    def validate_conditional_probability(self) -> "ConditionalTaskGraphEdge":
        """Validate conditional edge probability values."""
        if self.probability is not None and not (0.0 <= self.probability <= 1.0):
            raise ValueError("`probability` must be in the range [0, 1].")
        return self


class ConditionalTaskGraph(TaskGraph):
    """Task graph with helper methods for conditional scheduling."""

    def _is_conditional_edge(self, source: str, target: str) -> bool:
        """Check whether the edge (source -> target) is conditional."""
        for edge in self.dependencies:
            if (
                edge.source == source
                and edge.target == target
                and isinstance(edge, ConditionalTaskGraphEdge)
                and edge.probability < 1.0
            ):
                return True
        return False

    def _get_edge_probability(self, source: str, target: str) -> float:
        """Return the probability on a conditional edge, or 1.0 for deterministic."""
        for edge in self.dependencies:
            if edge.source == source and edge.target == target:
                if isinstance(edge, ConditionalTaskGraphEdge):
                    return edge.probability
                return 1.0
        return 1.0

    # ------------------------------------------------------------------
    # Trace discovery
    # ------------------------------------------------------------------

    def identify_traces(self) -> List[Tuple[str, List[str]]]:
        """Lightweight trace enumeration (name + task list only).

        See :meth:`identify_traces_detailed` for the full version that also
        computes trace probabilities and conditional edges used.
        """
        detailed = self.identify_traces_detailed()
        return [(trace["name"], trace["tasks"]) for trace in detailed]

    def identify_traces_detailed(self) -> List[dict]:
        """Enumerate every execution trace with probabilities.

        Uses recursive BFS. At each conditional fork the search splits,
        multiplying the running probability by the edge probability of the
        chosen child.

        Returns:
            List of dicts, each with keys:

            - ``name``: human-readable trace label
            - ``tasks``: sorted list of task names in this trace
            - ``probability``: product of conditional-edge probabilities
            - ``conditional_edges``: list of ``[parent, child]`` pairs
              used at each fork, useful for debugging

        Example::

            [
                {"name": "Trace: B-D", "tasks": ["A", "B", "D"],
                 "probability": 0.35,
                 "conditional_edges": [["A", "B"], ["B", "D"]]},
                ...
            ]
        """
        dag = self.graph

        def _explore(
            queue: Optional[List[str]] = None,
            visited: Optional[Set[str]] = None,
            path_names: Optional[List[str]] = None,
            probability: float = 1.0,
            cond_edges: Optional[List[List[str]]] = None,
        ) -> List[Tuple[List[str], Set[str], float, List[List[str]]]]:
            if queue is None:
                queue = [n for n in dag.nodes if dag.in_degree(n) == 0]
            if visited is None:
                visited = set()
            if path_names is None:
                path_names = []
            if cond_edges is None:
                cond_edges = []

            while queue:
                current = queue.pop(0)
                visited.add(current)

                children = list(dag.successors(current))
                conditional_children = [
                    c for c in children if self._is_conditional_edge(current, c)
                ]
                non_conditional_children = [
                    c for c in children if not self._is_conditional_edge(current, c)
                ]

                if conditional_children:
                    all_traces: List[
                        Tuple[List[str], Set[str], float, List[List[str]]]
                    ] = []
                    for child in conditional_children:
                        edge_prob = self._get_edge_probability(current, child)
                        new_queue = [child] + non_conditional_children + queue.copy()
                        all_traces.extend(
                            _explore(
                                new_queue,
                                visited.copy(),
                                path_names + [child],
                                probability * edge_prob,
                                cond_edges + [[current, child]],
                            )
                        )
                    return all_traces

                for child in children:
                    if child not in visited and child not in queue:
                        queue.append(child)

            return [(path_names, visited, probability, cond_edges)]

        raw = _explore()
        traces: List[dict] = []
        for path_names, visited, prob, cond_edges in raw:
            name = f"Trace: {'-'.join(path_names)}" if path_names else "No conditionals"
            traces.append(
                {
                    "name": name,
                    "tasks": sorted(visited),
                    "probability": round(prob, 6),
                    "conditional_edges": cond_edges,
                }
            )
        return traces

    def build_mutual_exclusion_graph(self) -> nx.Graph:
        """Build an undirected graph encoding mutual exclusion between tasks.

        Every task appears as a node. An edge between two tasks means they
        are mutually exclusive: they never execute together in any trace.

        This logic finds every possible task pair in the task graph, then
        records all pairs that co-occur in at least one trace. Any pair that
        never co-occurs is mutually exclusive.

        Returns:
            ``nx.Graph`` whose edges represent mutually exclusive task pairs.
        """
        traces = self.identify_traces()
        all_tasks = {task.name for task in self.tasks}

        co_occurring: Set[frozenset] = set()
        for _name, trace_tasks in traces:
            trace_set = set(trace_tasks)
            for a, b in combinations(trace_set, 2):
                co_occurring.add(frozenset((a, b)))

        meg = nx.Graph()
        meg.add_nodes_from(all_tasks)

        for a, b in combinations(all_tasks, 2):
            if frozenset((a, b)) not in co_occurring:
                meg.add_edge(a, b)

        return meg


def extract_trace_schedules(
    schedule: Schedule,
) -> Dict[str, Dict[str, List[ScheduledTask]]]:
    """Split an overlapping conditional schedule into per-trace schedules.

    For each trace identified in the task graph, returns a copy of the
    schedule mapping containing only the tasks that belong to that trace.

    Args:
        schedule: The overlapping schedule produced by any scheduler.

    Example returned schedule mapping:
        {
        "Trace: B-D": {"n0": [A, D], "n1": [B]},
        "Trace: C-F": {"n0": [A, F], "n1": [C]},
        }
    """
    task_graph = schedule.task_graph
    if not isinstance(task_graph, ConditionalTaskGraph):
        raise TypeError(
            "extract_trace_schedules requires a schedule over a "
            "ConditionalTaskGraph."
        )
    traces = task_graph.identify_traces()

    trace_schedules: Dict[str, Dict[str, List[ScheduledTask]]] = {}
    for trace_name, trace_tasks in traces:
        task_set = set(trace_tasks)
        trace_mapping: Dict[str, List[ScheduledTask]] = {}
        for node_name, tasks in schedule.mapping.items():
            trace_mapping[node_name] = [
                ScheduledTask(node=t.node, name=t.name, start=t.start, end=t.end)
                for t in tasks
                if t.name in task_set
            ]
        trace_schedules[trace_name] = trace_mapping

    return trace_schedules


def recalculate_trace_times(
    trace_mapping: Dict[str, List[ScheduledTask]],
    schedule: Schedule,
) -> Dict[str, List[ScheduledTask]]:
    """Recalculate start/end times for a trace schedule to remove gaps.

    Keeps the same node assignments but walks tasks in topological order
    and computes the earliest feasible start based on predecessors that
    actually exist in this trace.

    Args:
        trace_mapping: Per-node task lists for one trace, possibly with gaps.
        schedule: The original overlapping schedule, which provides task graph
            and network data for cost and speed lookups.

    Returns:
        A new mapping ``{node_name: [ScheduledTask, ...]}`` with tight times.
    """
    task_graph = schedule.task_graph
    network = schedule.network

    task_info: Dict[str, Tuple[str, ScheduledTask]] = {}
    for node_name, tasks in trace_mapping.items():
        for scheduled_task in tasks:
            task_info[scheduled_task.name] = (node_name, scheduled_task)

    end_times: Dict[str, float] = {}
    new_mapping: Dict[str, List[ScheduledTask]] = {n: [] for n in trace_mapping}
    node_available: Dict[str, float] = {n: 0.0 for n in trace_mapping}

    for task_name in nx.topological_sort(task_graph.graph):
        if task_name not in task_info:
            continue

        node_name, _original = task_info[task_name]
        node = network.get_node(node_name)
        exec_time = task_graph.get_task(task_name).cost / node.speed

        predecessors = [
            p for p in task_graph.graph.predecessors(task_name) if p in task_info
        ]
        if predecessors:
            dep_ready = max(
                end_times[p]
                + _comm_time(
                    p,
                    task_info[p][0],
                    task_name,
                    node_name,
                    task_graph,
                    network,
                )
                for p in predecessors
            )
        else:
            dep_ready = 0.0

        start = max(dep_ready, node_available[node_name])
        end = start + exec_time
        end_times[task_name] = end
        node_available[node_name] = end
        new_mapping[node_name].append(
            ScheduledTask(node=node_name, name=task_name, start=start, end=end)
        )

    for node_tasks in new_mapping.values():
        node_tasks.sort(key=lambda t: t.start)

    return new_mapping


def extract_traces_with_recalculation(
    schedule: Schedule,
) -> Dict[str, Dict[str, List[ScheduledTask]]]:
    """Extract per-trace schedules and recalculate times to remove gaps."""
    trace_schedules = extract_trace_schedules(schedule)
    return {
        name: recalculate_trace_times(mapping, schedule)
        for name, mapping in trace_schedules.items()
    }


def schedule_trace_standalone(
    trace_tasks: List[str],
    task_graph: ConditionalTaskGraph,
    network: Network,
    scheduler: Scheduler,
) -> Schedule:
    """Schedule a trace as a standalone non-conditional task graph.

    Extracts the subgraph for the given trace tasks, converts it to a plain
    TaskGraph, stripping conditional metadata, and schedules it from scratch
    with the given heuristic.

    Args:
        trace_tasks: Task names belonging to this trace.
        task_graph: The original conditional task graph.
        network: The network to schedule on.
        scheduler: The heuristic to use.

    Returns:
        A fresh Schedule for just this trace's tasks.
    """
    trace_subgraph = nx.DiGraph(task_graph.graph.subgraph(trace_tasks))
    standalone_tg = TaskGraph.from_nx(trace_subgraph)
    return scheduler.schedule(network=network, task_graph=standalone_tg)


def _comm_time(
    src_task: str,
    src_node: str,
    dst_task: str,
    dst_node: str,
    task_graph: TaskGraph,
    network: Network,
) -> float:
    """Communication cost between two tasks on possibly different nodes."""
    if src_node == dst_node:
        return 0.0
    edge = network.get_edge(src_node, dst_node)
    dep = task_graph.get_dependency(src_task, dst_task)
    if edge.speed == 0.0:
        return 0.0
    return dep.size / edge.speed
