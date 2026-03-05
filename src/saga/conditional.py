"""Conditional task graph primitives.

This module intentionally contains only two core pieces:
- `ConditionalTaskGraphEdge`: edge metadata for conditional branches.
- `ConditionalTaskGraph`: graph helper(s), including group discovery.
"""

from typing import Dict, Optional
from abc import ABC, abstractmethod
import bisect
import logging
import math
from functools import cached_property
from itertools import product
from queue import PriorityQueue
from typing import Dict, FrozenSet, Generator, Iterable, List, Optional, Set, Tuple

import networkx as nx
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from saga.utils.random_variable import RandomVariable, DEFAULT_NUM_SAMPLES
from saga import (
    Schedule,
    ScheduledTask,
    TaskGraph,
    Network,
    NetworkNode,
    NetworkEdge,
    TaskGraphNode,
    TaskGraphEdge,
)


class ConditionalTaskGraphEdge(TaskGraphEdge):
    """Task graph edge with conditional branch metadata."""

    conditional: bool = Field(
        default=False,
        description="Whether this dependency edge is a conditional branch.",
    )
    probability: Optional[float] = Field(
        default=None,
        description="Branch probability in [0, 1] when `conditional=True`.",
    )

    @model_validator(mode="after")
    def validate_conditional_probability(self) -> "ConditionalTaskGraphEdge":
        """Validate conditional edge probability values."""
        if self.conditional and self.probability is None:
            raise ValueError(
                "Conditional edges must define `probability` in the range [0, 1]."
            )
        if self.probability is not None and not (0.0 <= self.probability <= 1.0):
            raise ValueError("`probability` must be in the range [0, 1].")
        return self


class ConditionalTaskGraph(TaskGraph):
    """Task graph with helper methods for conditional scheduling."""

    def identify_conditional_groups(self) -> Dict[str, int]:
        """Return task -> group id mapping for conditional alternatives.

        Grouping rule:
        - If a parent has multiple outgoing conditional edges, those child tasks
          are alternatives and are assigned the same group id.
        - Tasks that are not part of any conditional-alternative set get `-1`.

        Example output (A -> B, A -> C conditionally; B,C -> D):
            {"A": -1, "B": 0, "C": 0, "D": -1}
        """
        groups: Dict[str, int] = {}
        next_group_id = 0
        task_names = {task.name for task in self.tasks}

        for parent_name in task_names:
            conditional_children = [
                edge.target
                for edge in self.dependencies
                if edge.source == parent_name
                and isinstance(edge, ConditionalTaskGraphEdge)
                and edge.conditional
            ]
            if len(conditional_children) > 1:
                for child_name in conditional_children:
                    groups[child_name] = next_group_id
                next_group_id += 1

        for task_name in task_names:
            groups.setdefault(task_name, -1)

        return groups


