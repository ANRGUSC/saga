"""Helpers for scheduling conditional task graphs (CTGs).

When scheduling a CTG:
1) Identify all individual execution branches.
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
    ScheduledTask,
    TaskGraph,
    TaskGraphEdge,
)


class ConditionalTaskGraphEdge(TaskGraphEdge):
    """Task graph edge with conditional branch metadata."""

    probability: float = Field(
        default=1.0,
        description="Branch probability in [0, 1], assume conditionality if probability set to < 1.",
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

    def identify_branches(self) -> List[Tuple[str, List[str]]]:
        """Identify all possible execution branches in the conditional task graph.
        
        Uses a recursive BFS-like approach similar to bfs.py to handle multiple branching points.
        
        Args:
            task_graph: The conditional task graph
            
        Returns:
            List of (branch_name, tasks_in_branch) tuples
            
        Example:
            For A -> B/C -> D where both B and C lead to D:
            Returns: [("Path: B", ["A", "B", "D"]), ("Path: C", ["A", "C", "D"])]
            
            For A -> B/C, B -> D/E (multiple branches):
            Returns: [("Path: B-D", ["A", "B", "D"]), 
                    ("Path: B-E", ["A", "B", "E"]),
                    ("Path: C", ["A", "C"])]
        """
        dag = self.graph

        def _explore(
            queue: Optional[List[str]] = None,
            visited: Optional[Set[str]] = None,
            path_names: Optional[List[str]] = None,
        ) -> List[Tuple[List[str], Set[str]]]:
            if queue is None:
                queue = [n for n in dag.nodes if dag.in_degree(n) == 0]
            if visited is None:
                visited = set()
            if path_names is None:
                path_names = []

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
                    all_branches: List[Tuple[List[str], Set[str]]] = []
                    for child in conditional_children:
                        new_queue = [child] + non_conditional_children + queue.copy()
                        all_branches.extend(
                            _explore(new_queue, visited.copy(), path_names + [child])
                        )
                    return all_branches
                else:
                    for child in children:
                        if child not in visited and child not in queue:
                            queue.append(child)

            return [(path_names, visited)]

        raw = _explore()
        branches: List[Tuple[str, List[str]]] = []
        for path_names, visited in raw:
            name = f"Path: {'-'.join(path_names)}" if path_names else "No conditionals"
            branches.append((name, sorted(visited)))
        return branches

    def build_mutual_exclusion_graph(self) -> nx.Graph:
        """Build an undirected graph encoding mutual exclusion between tasks.

        Every task appears as a node.  An edge between two tasks means they
        are **mutually exclusive** (never execute together in any branch).

        This logic is simple, it finds every possible task pair in the taskgraph
        then it goes through each branch and creates all possible task pairs

        then it compares the task pairs found in branches and all existing task pairs, 
        if there is a task pair that did not exist when going through branches then 
        it means the tasks are not mutually exclusive and we draw a line between them

        Returns:
            ``nx.Graph`` whose edges represent mutually exclusive task pairs.
        """
        branches = self.identify_branches()
        all_tasks = {task.name for task in self.tasks}

        co_occurring: Set[frozenset] = set()
        for _name, branch_tasks in branches:
            branch_set = set(branch_tasks)
            for a, b in combinations(branch_set, 2):
                co_occurring.add(frozenset((a, b)))

        meg = nx.Graph()
        meg.add_nodes_from(all_tasks)

        for a, b in combinations(all_tasks, 2):
            if frozenset((a, b)) not in co_occurring:
                meg.add_edge(a, b)

        return meg


def extract_branch_schedules(
    schedule: Schedule,
) -> Dict[str, Dict[str, List[ScheduledTask]]]:
    """Split an overlapping conditional schedule into per-branch schedules.

    For each branch identified in the task graph, returns a copy of the
    schedule mapping containing only the tasks that belong to that branch.

    Args:
        schedule: The overlapping schedule produced by any scheduler.

    Example return of retuned scheudule mapping:
        {
        "Path: B-D": {"n0": [A, D], "n1": [B]},
        "Path: C-F": {"n0": [A, F], "n1": [C]},
        }
    """
    task_graph: ConditionalTaskGraph = schedule.task_graph  
    branches = task_graph.identify_branches()

    branch_schedules: Dict[str, Dict[str, List[ScheduledTask]]] = {}
    #loop through all branches
    for branch_name, branch_tasks in branches:
        task_set = set(branch_tasks)
        branch_mapping: Dict[str, List[ScheduledTask]] = {}
        #loop through all tasks in schedule mapping and if current task not in current branch then skip it otherwise add it to new mapping
        for node_name, tasks in schedule.mapping.items():
            branch_mapping[node_name] = [
                ScheduledTask(node=t.node, name=t.name, start=t.start, end=t.end)
                for t in tasks
                if t.name in task_set
            ]
        branch_schedules[branch_name] = branch_mapping

    return branch_schedules


def recalculate_branch_times(
    branch_mapping: Dict[str, List[ScheduledTask]],
    schedule: Schedule,
) -> Dict[str, List[ScheduledTask]]:
    """Recalculate start/end times for a branch schedule to remove gaps.

    Keeps the same node assignments but walks tasks in topological order
    and computes the earliest feasible start based on predecessors that
    actually exist in this branch.

    Args:
        branch_mapping: Per-node task lists for one branch (may have gaps).
        schedule: The original overlapping schedule (provides task graph &
            network for cost / speed lookups).

    Returns:
        A new mapping ``{node_name: [ScheduledTask, ...]}`` with tight times.
    """
    task_graph = schedule.task_graph
    network = schedule.network

    task_info: Dict[str, Tuple[str, ScheduledTask]] = {}
    for node_name, tasks in branch_mapping.items():
        for t in tasks:
            task_info[t.name] = (node_name, t)

    end_times: Dict[str, float] = {}
    new_mapping: Dict[str, List[ScheduledTask]] = {n: [] for n in branch_mapping}

    for task_name in nx.topological_sort(task_graph.graph):
        if task_name not in task_info:
            continue

        node_name, original = task_info[task_name]
        node = network.get_node(node_name)
        exec_time = task_graph.get_task(task_name).cost / node.speed

        predecessors = [
            p for p in task_graph.graph.predecessors(task_name) if p in task_info
        ]
        if predecessors:
            start = max(
                end_times[p] + _comm_time(p, task_info[p][0], task_name, node_name, task_graph, network)
                for p in predecessors
            )
        else:
            start = 0.0

        end = start + exec_time
        end_times[task_name] = end
        new_mapping[node_name].append(
            ScheduledTask(node=node_name, name=task_name, start=start, end=end)
        )

    for node_tasks in new_mapping.values():
        node_tasks.sort(key=lambda t: t.start)

    return new_mapping


def extract_branches_with_recalculation(
    schedule: Schedule,
) -> Dict[str, Dict[str, List[ScheduledTask]]]:
    """
    Extract per-branch schedules and recalculate times to remove gaps.
    """
    branch_schedules = extract_branch_schedules(schedule)
    return {
        name: recalculate_branch_times(mapping, schedule)
        for name, mapping in branch_schedules.items()
    }


def _comm_time(
    src_task: str, src_node: str,
    dst_task: str, dst_node: str,
    task_graph: TaskGraph, network: Network,
) -> float:
    """Communication cost between two tasks on (possibly different) nodes."""
    if src_node == dst_node:
        return 0.0
    edge = network.get_edge(src_node, dst_node)
    dep = task_graph.get_dependency(src_task, dst_task)
    if edge.speed == 0.0:
        return 0.0
    return dep.size / edge.speed
