"""Overlap policies: decide whether two tasks may share a node/time slot.

A :class:`~saga.Schedule` normally forbids two tasks from occupying the same
processor at the same time. Some task graphs relax that: in a conditional task
graph, tasks on mutually exclusive branches never execute together, so a
scheduler may safely overlap them on one node.

That relaxation is expressed as an :class:`OverlapPolicy`, a small strategy the
``Schedule`` consults instead of hard-coding any single notion of overlap:

- :class:`NoOverlapPolicy` (the default) never allows overlap: classic scheduling.
- :class:`AllowOverlapPolicy` allows overlap between an explicit set of task pairs.

Overlap is a *symmetric pairwise relation*, not a flat set of tasks: task ``A``
may overlap ``B`` and ``B`` may overlap ``C`` without ``A`` being allowed to
overlap ``C`` (they can co-occur in some execution). ``AllowOverlapPolicy`` therefore
stores pairs, not a single set.
"""

from typing import Dict, FrozenSet, Iterable, Protocol, Set, Tuple, runtime_checkable

_EMPTY: FrozenSet[str] = frozenset()


@runtime_checkable
class OverlapPolicy(Protocol):
    """Decides whether two tasks are permitted to overlap in time on one node."""

    def can_overlap(self, task_a: str, task_b: str) -> bool:
        """Return True if ``task_a`` and ``task_b`` may occupy the same slot."""
        ...


class NoOverlapPolicy:
    """Default policy: no two tasks may overlap (classic scheduling)."""

    def can_overlap(self, task_a: str, task_b: str) -> bool:
        return False


class AllowOverlapPolicy:
    """Permits overlap between explicitly listed pairs of tasks.

    The relation is stored as an adjacency map (``task -> {tasks it may overlap}``)
    so that :meth:`can_overlap` is O(1) with no per-query allocation. Use
    :meth:`from_pairs` to build one; it enforces symmetry.
    """

    def __init__(self, adjacency: Dict[str, FrozenSet[str]]) -> None:
        self._adjacency = adjacency

    @classmethod
    def from_pairs(cls, pairs: Iterable[Tuple[str, str]]) -> "AllowOverlapPolicy":
        """Build a policy from unordered task pairs that may overlap.

        Args:
            pairs: Iterable of ``(task_a, task_b)`` pairs. Order within a pair
                does not matter; both directions are recorded. Self-pairs are
                ignored.

        Returns:
            An :class:`AllowOverlapPolicy` permitting exactly those pairs.
        """
        adjacency: Dict[str, Set[str]] = {}
        for task_a, task_b in pairs:
            if task_a == task_b:
                continue
            adjacency.setdefault(task_a, set()).add(task_b)
            adjacency.setdefault(task_b, set()).add(task_a)
        return cls({task: frozenset(neighbors) for task, neighbors in adjacency.items()})

    def can_overlap(self, task_a: str, task_b: str) -> bool:
        return task_b in self._adjacency.get(task_a, _EMPTY)
