import abc
import random
from typing import Annotated, Any, Literal, Optional, Set, Union
from pydantic import BaseModel, Discriminator, Field, Tag

from saga import (
    Network,
    TaskGraph,
    NetworkNode,
    NetworkEdge,
    TaskGraphNode,
    TaskGraphEdge,
)

MINVAL = 0.1
MAXVAL = 1.0
DELTA = 0.1


class Change(BaseModel, abc.ABC):
    """Base class for changes to a network or task graph."""

    @classmethod
    def random(cls, network: Network, task_graph: TaskGraph) -> Optional["Change"]:
        """Generate a random change.

        Args:
            network: The network.
            task_graph: The task graph.

        Returns:
            A random change, or None if no valid change can be made.
        """
        raise NotImplementedError

    @classmethod
    def apply_random(
        cls, network: Network, task_graph: TaskGraph
    ) -> tuple[Optional["Change"], Network, TaskGraph]:
        """Apply a random change to the network and task graph.

        Args:
            network: The network.
            task_graph: The task graph.

        Returns:
            Tuple of (change, new_network, new_task_graph).
        """
        change: Optional[Change] = cls.random(network, task_graph)
        if change is not None:
            network, task_graph = change.apply(network, task_graph)
        return change, network, task_graph

    @abc.abstractmethod
    def apply(
        self, network: Network, task_graph: TaskGraph
    ) -> tuple[Network, TaskGraph]:
        """Apply the change to the network and task graph.

        Args:
            network: The network.
            task_graph: The task graph.

        Returns:
            Tuple of (new_network, new_task_graph).
        """
        raise NotImplementedError


class TaskGraphDeleteDependency(Change):
    """Delete a dependency from the task graph."""

    change_type: Literal["task_graph_delete_dependency"] = (
        "task_graph_delete_dependency"
    )
    source: str = Field(..., description="The source task.")
    target: str = Field(..., description="The target task.")

    @classmethod
    def random(
        cls, network: Network, task_graph: TaskGraph
    ) -> Optional["TaskGraphDeleteDependency"]:
        """Randomly select a dependency to delete."""
        if len(task_graph.dependencies) <= 0:
            return None
        dep = random.choice(list(task_graph.dependencies))
        return cls(source=dep.source, target=dep.target)

    def apply(
        self, network: Network, task_graph: TaskGraph
    ) -> tuple[Network, TaskGraph]:
        new_deps = frozenset(
            d
            for d in task_graph.dependencies
            if not (d.source == self.source and d.target == self.target)
        )
        new_task_graph = TaskGraph(tasks=task_graph.tasks, dependencies=new_deps)
        return network, new_task_graph


class TaskGraphAddDependency(Change):
    """Add a dependency to the task graph."""

    change_type: Literal["task_graph_add_dependency"] = "task_graph_add_dependency"
    source: str = Field(..., description="The source task.")
    target: str = Field(..., description="The target task.")
    size: float = Field(
        default_factory=lambda: random.uniform(MINVAL, MAXVAL),
        description="The data size.",
    )

    @classmethod
    def random(
        cls, network: Network, task_graph: TaskGraph
    ) -> Optional["TaskGraphAddDependency"]:
        """Randomly select a dependency to add."""
        import networkx as nx

        graph = task_graph.graph

        nodes = list(graph.nodes)
        random.shuffle(nodes)

        for node in nodes:
            ancestors = nx.ancestors(graph, node)
            children = set(graph.successors(node))
            non_ancestors = set(nodes) - ancestors - children - {node}
            if len(non_ancestors) > 0:
                target = random.choice(list(non_ancestors))
                return cls(source=node, target=target)
        return None

    def apply(
        self, network: Network, task_graph: TaskGraph
    ) -> tuple[Network, TaskGraph]:
        new_dep = TaskGraphEdge(source=self.source, target=self.target, size=self.size)
        new_deps = frozenset(task_graph.dependencies | {new_dep})
        new_task_graph = TaskGraph(tasks=task_graph.tasks, dependencies=new_deps)
        return network, new_task_graph


class TaskGraphChangeDependencyWeight(Change):
    """Change a dependency weight in the task graph."""

    change_type: Literal["task_graph_change_dependency_weight"] = (
        "task_graph_change_dependency_weight"
    )
    source: str = Field(..., description="The source task.")
    target: str = Field(..., description="The target task.")
    size: float = Field(..., description="The new data size.")

    @classmethod
    def random(
        cls, network: Network, task_graph: TaskGraph
    ) -> Optional["TaskGraphChangeDependencyWeight"]:
        """Randomly select a dependency to change."""
        if len(task_graph.dependencies) <= 0:
            return None
        dep = random.choice(list(task_graph.dependencies))
        d_size = random.uniform(-DELTA, DELTA)
        new_size = max(MINVAL, min(MAXVAL, dep.size + d_size))
        return cls(source=dep.source, target=dep.target, size=new_size)

    def apply(
        self, network: Network, task_graph: TaskGraph
    ) -> tuple[Network, TaskGraph]:
        new_deps: Set[TaskGraphEdge] = set()
        for d in task_graph.dependencies:
            if d.source == self.source and d.target == self.target:
                new_deps.add(
                    TaskGraphEdge(source=d.source, target=d.target, size=self.size)
                )
            else:
                new_deps.add(d)
        new_task_graph = TaskGraph(
            tasks=task_graph.tasks, dependencies=frozenset(new_deps)
        )
        return network, new_task_graph


class TaskGraphChangeTaskWeight(Change):
    """Change a task weight in the task graph."""

    change_type: Literal["task_graph_change_task_weight"] = (
        "task_graph_change_task_weight"
    )
    task: str = Field(..., description="The task name.")
    cost: float = Field(..., description="The new cost.")

    @classmethod
    def random(
        cls, network: Network, task_graph: TaskGraph
    ) -> Optional["TaskGraphChangeTaskWeight"]:
        """Randomly select a task to change."""
        if len(task_graph.tasks) <= 0:
            return None
        task = random.choice(list(task_graph.tasks))
        d_cost = random.uniform(-DELTA, DELTA)
        new_cost = max(MINVAL, min(MAXVAL, task.cost + d_cost))
        return cls(task=task.name, cost=new_cost)

    def apply(
        self, network: Network, task_graph: TaskGraph
    ) -> tuple[Network, TaskGraph]:
        new_tasks: Set[TaskGraphNode] = set()
        for t in task_graph.tasks:
            if t.name == self.task:
                new_tasks.add(TaskGraphNode(name=t.name, cost=self.cost))
            else:
                new_tasks.add(t)
        new_task_graph = TaskGraph(
            tasks=frozenset(new_tasks), dependencies=task_graph.dependencies
        )
        return network, new_task_graph


class NetworkChangeEdgeWeight(Change):
    """Change a network edge weight."""

    change_type: Literal["network_change_edge_weight"] = "network_change_edge_weight"
    source: str = Field(..., description="The source node.")
    target: str = Field(..., description="The target node.")
    speed: float = Field(..., description="The new speed (bandwidth).")

    @classmethod
    def random(
        cls, network: Network, task_graph: TaskGraph
    ) -> Optional["NetworkChangeEdgeWeight"]:
        """Randomly select an edge to change."""
        # Filter out self-loops
        non_self_edges = [e for e in network.edges if e.source != e.target]
        if len(non_self_edges) <= 0:
            return None
        edge = random.choice(non_self_edges)
        d_speed = random.uniform(-DELTA, DELTA)
        new_speed = max(MINVAL, min(MAXVAL, edge.speed + d_speed))
        return cls(source=edge.source, target=edge.target, speed=new_speed)

    def apply(
        self, network: Network, task_graph: TaskGraph
    ) -> tuple[Network, TaskGraph]:
        new_edges: Set[NetworkEdge] = set()
        for e in network.edges:
            src, tgt = sorted((e.source, e.target))
            self_src, self_tgt = sorted((self.source, self.target))
            if src == self_src and tgt == self_tgt:
                new_edges.add(
                    NetworkEdge(source=e.source, target=e.target, speed=self.speed)
                )
            else:
                new_edges.add(e)
        new_network = Network(nodes=network.nodes, edges=frozenset(new_edges))
        return new_network, task_graph


class NetworkChangeNodeWeight(Change):
    """Change a network node weight."""

    change_type: Literal["network_change_node_weight"] = "network_change_node_weight"
    node: str = Field(..., description="The node name.")
    speed: float = Field(..., description="The new speed.")

    @classmethod
    def random(
        cls, network: Network, task_graph: TaskGraph
    ) -> Optional["NetworkChangeNodeWeight"]:
        """Randomly select a node to change."""
        if len(network.nodes) <= 0:
            return None
        node = random.choice(list(network.nodes))
        d_speed = random.uniform(-DELTA, DELTA)
        new_speed = max(MINVAL, min(MAXVAL, node.speed + d_speed))
        return cls(node=node.name, speed=new_speed)

    def apply(
        self, network: Network, task_graph: TaskGraph
    ) -> tuple[Network, TaskGraph]:
        new_nodes: Set[NetworkNode] = set()
        for n in network.nodes:
            if n.name == self.node:
                new_nodes.add(NetworkNode(name=n.name, speed=self.speed))
            else:
                new_nodes.add(n)
        new_network = Network(nodes=frozenset(new_nodes), edges=network.edges)
        return new_network, task_graph


# Discriminator function for Change union type
def get_change_discriminator_value(v: Any) -> str:
    """Get the discriminator value for a Change instance or dict."""
    if isinstance(v, dict):
        return v.get("change_type", "")
    return getattr(v, "change_type", "")


# Discriminated union type for all Change types
ChangeType = Annotated[
    Union[
        Annotated[TaskGraphDeleteDependency, Tag("task_graph_delete_dependency")],
        Annotated[TaskGraphAddDependency, Tag("task_graph_add_dependency")],
        Annotated[
            TaskGraphChangeDependencyWeight, Tag("task_graph_change_dependency_weight")
        ],
        Annotated[TaskGraphChangeTaskWeight, Tag("task_graph_change_task_weight")],
        Annotated[NetworkChangeEdgeWeight, Tag("network_change_edge_weight")],
        Annotated[NetworkChangeNodeWeight, Tag("network_change_node_weight")],
    ],
    Discriminator(get_change_discriminator_value),
]

# Default change types for simulated annealing
DEFAULT_CHANGE_TYPES = [
    TaskGraphDeleteDependency,
    TaskGraphAddDependency,
    TaskGraphChangeDependencyWeight,
    TaskGraphChangeTaskWeight,
    NetworkChangeEdgeWeight,
    NetworkChangeNodeWeight,
]
