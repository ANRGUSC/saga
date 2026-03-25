"""Conditional task graph primitives.

This module contains:
- `ConditionalTaskGraphEdge`: edge metadata for conditional branches.
- `ConditionalTaskGraph`: graph helper(s), including BFS branch extraction
  and mutual-exclusion graph construction.
"""

from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from pydantic import Field, model_validator

from saga import (
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
        """Enumerate every possible execution branch via recursive BFS.

        At each conditional branching point the search forks: one sub-path
        per conditional child.  Non-conditional children are always included.

        Returns:
            List of ``(branch_name, tasks_in_branch)`` tuples.
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
        If two tasks can co-occur in at least one branch, there is no edge.

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


