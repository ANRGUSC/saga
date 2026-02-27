from abc import ABC, abstractmethod
from typing import Any, List

from pydantic import BaseModel, Field


class Constraint(BaseModel, ABC):
    """Abstract base for a placement constraint on a single task."""

    task: str = Field(..., description="The name of the task this constraint applies to.")

    @abstractmethod
    def allowed_nodes(self, node_names: List[str]) -> List[str]:
        """Return the subset of node_names this task is allowed to run on.

        Args:
            node_names: All node names present in the network.

        Returns:
            List[str]: Node names the task may be placed on.
        """
        ...


class AllowedNodes(Constraint):
    """Allow-list constraint: the task may only run on the specified nodes.

    This is a strict generalisation of the existing ``pinned_to`` field —
    a single-node pin is expressed as::

        AllowedNodes(task="t_1", nodes=["v_1"])
    """

    nodes: List[str] = Field(..., description="Nodes this task is allowed to run on.")

    def allowed_nodes(self, node_names: List[str]) -> List[str]:
        return [n for n in self.nodes if n in node_names]


class ForbiddenNodes(Constraint):
    """Exclude-list constraint: the task must NOT run on the specified nodes.

    This mirrors the ``schedule_restrictions`` model used in the earlier
    research version of SAGA, but expressed as a proper first-class object
    rather than a graph-level attribute.
    """

    nodes: List[str] = Field(..., description="Nodes this task may not run on.")

    def allowed_nodes(self, node_names: List[str]) -> List[str]:
        forbidden = set(self.nodes)
        return [n for n in node_names if n not in forbidden]


class Constraints(BaseModel):
    """A collection of placement constraints for a scheduling problem.

    Pass a ``Constraints`` instance to any scheduler's ``schedule()`` method
    to restrict which network nodes each task may be assigned to.

    Supported constraint types:

    * :class:`AllowedNodes` — task may only run on a specified set of nodes
      (generalises the ``pinned_to`` field; a single-node pin is a special case).
    * :class:`ForbiddenNodes` — task must not run on a specified set of nodes
      (mirrors the old ``schedule_restrictions`` graph-attribute approach).

    Example::

        from saga.constraints import AllowedNodes, ForbiddenNodes, Constraints

        constraints = Constraints(constraints=[
            AllowedNodes(task="t_1", nodes=["v_1"]),          # pin t_1 to v_1
            AllowedNodes(task="t_2", nodes=["v_1", "v_2"]),   # restrict t_2
            ForbiddenNodes(task="t_3", nodes=["v_3"]),         # forbid t_3 on v_3
        ])
        schedule = scheduler.schedule(network, task_graph, constraints=constraints)
    """

    constraints: List[Constraint] = Field(
        default_factory=list,
        description="The list of placement constraints.",
    )

    def get_candidate_nodes(self, task: str, node_names: List[str]) -> List[str]:
        """Return the nodes this task is allowed to run on.

        If no constraint is registered for ``task``, all nodes are returned.

        Args:
            task: The task name.
            node_names: All node names present in the network.

        Returns:
            List[str]: Candidate node names for the task (always non-empty).

        Raises:
            ValueError: If no nodes remain after applying the constraint.
        """
        for constraint in self.constraints:
            if constraint.task == task:
                candidates = constraint.allowed_nodes(node_names)
                if not candidates:
                    raise ValueError(
                        f"Constraint for task '{task}' yields no valid nodes. "
                        f"Verify that the specified nodes exist in the network."
                    )
                return candidates
        return list(node_names)

    @classmethod
    def from_task_graph(cls, task_graph: Any) -> "Constraints":
        """Build a :class:`Constraints` object from legacy ``pinned_to`` fields.

        Provides backward compatibility: any task whose ``pinned_to`` attribute
        is set is automatically wrapped in an :class:`AllowedNodes` constraint.

        Args:
            task_graph: A ``TaskGraph`` instance.

        Returns:
            Constraints: Extracted constraints, or an empty instance if none.
        """
        items: List[Constraint] = []
        for task in task_graph.tasks:
            if task.pinned_to is not None:
                items.append(AllowedNodes(task=task.name, nodes=[task.pinned_to]))
        return cls(constraints=items)
