from abc import ABC, abstractmethod
import bisect
import logging
import math
from functools import cached_property
from itertools import product
from queue import PriorityQueue
from typing import Dict, FrozenSet, Generator, Iterable, List, Optional, Set, Tuple

import networkx as nx
from pydantic import BaseModel, Field, PrivateAttr

from saga.stochastic import StochasticTaskGraph, StochasticTaskGraphNode
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

class ConditionalTaskGraphNode(StochasticTaskGraphNode):
    """Represents a task that can have conditional branches."""

    name: str
    conditional_branches: List[Tuple[StochasticTaskGraph, float]] = Field(
        default_factory=list,
        description="List of (task, probability) pairs representing conditional branches.",
    )

    def __init__(self, name: str, conditional_branches: List[Tuple[StochasticTaskGraph, float]], **data):
        cost_by_branch = [
            sum(task.cost for task in branch.tasks) for branch, _ in conditional_branches
        ]
        num_samples = max(len(branch.tasks) for branch, _ in conditional_branches) * 10

        samples = []
        for cost in cost_by_branch:
            if isinstance(cost, RandomVariable):
                samples.extend(cost.samples)
            else:
                samples.extend([cost] * num_samples)
        cost = RandomVariable(samples=samples)

        super().__init__(name=name, cost=cost, **data)
        total_prob = sum(prob for _, prob in self.conditional_branches)
        if total_prob != 1.0:
            raise ValueError("Total probability of conditional branches must sum to 1.0")


class ConditionalTaskGraph(StochasticTaskGraph):
    """A TaskGraph that supports conditional branches."""

    def __init__(
        self,
        nodes: List[StochasticTaskGraphNode | ConditionalTaskGraphNode],
        edges: List[TaskGraphEdge],
        condition_edges: List[TaskGraphEdge],
    ):
        pass # TODO: What to do...