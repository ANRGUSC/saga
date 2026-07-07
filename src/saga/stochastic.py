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

from saga.utils.random_variable import RandomVariable, DEFAULT_NUM_SAMPLES
from saga import (
    ConstraintViolation,
    Schedule,
    ScheduledTask,
    TaskGraph,
    Network,
    NetworkNode,
    NetworkEdge,
    TaskGraphNode,
    TaskGraphEdge,
)


class StochasticNetworkNode(BaseModel):
    """A node in the network with stochastic speed."""

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    name: str = Field(..., description="The name of the node.")
    speed: RandomVariable = Field(..., description="The speed of the node.")

    def __hash__(self) -> int:
        return hash(self.name)

    def __lt__(self, other: "StochasticNetworkNode") -> bool:
        return self.name < other.name

    def sample(self) -> NetworkNode:
        """Sample the stochastic node to get a deterministic node."""
        return NetworkNode(name=self.name, speed=self.speed.sample(num_samples=1)[0])


class StochasticNetworkEdge(BaseModel):
    """An edge in the network with stochastic speed."""

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    source: str = Field(..., description="The source node of the edge.")
    target: str = Field(..., description="The target node of the edge.")
    speed: RandomVariable = Field(..., description="The speed (bandwidth) of the edge.")

    def __hash__(self) -> int:
        return hash((self.source, self.target))

    def __lt__(self, other: "StochasticNetworkEdge") -> bool:
        return (self.source, self.target) < (other.source, other.target)

    def sample(self) -> NetworkEdge:
        """Sample the stochastic edge to get a deterministic edge."""
        return NetworkEdge(
            source=self.source,
            target=self.target,
            speed=self.speed.sample(num_samples=1)[0],
        )


class StochasticNetwork(BaseModel):
    """A network of nodes and edges with stochastic values."""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    nodes: FrozenSet[StochasticNetworkNode] = Field(
        ..., description="The nodes in the network."
    )
    edges: FrozenSet[StochasticNetworkEdge] = Field(
        ..., description="The edges in the network."
    )

    @cached_property
    def computed_hash(self) -> int:
        return hash((frozenset(self.nodes), frozenset(self.edges)))

    def __hash__(self) -> int:
        return self.computed_hash

    def sample(self) -> Network:
        """Sample the stochastic network to get a deterministic network.

        Sorted iteration keeps the np.random draw order (and thus the realization)
        independent of PYTHONHASHSEED.
        """
        return Network.create(
            nodes=[node.sample() for node in sorted(self.nodes, key=lambda n: n.name)],
            edges=[
                edge.sample()
                for edge in sorted(self.edges, key=lambda e: (e.source, e.target))
            ],
        )

    @classmethod
    def create(
        cls,
        nodes: Iterable[StochasticNetworkNode | Tuple[str, RandomVariable | float]],
        edges: Iterable[
            StochasticNetworkEdge
            | Tuple[str, str, RandomVariable | float]
            | Tuple[str, str]
        ],
    ) -> "StochasticNetwork":
        """Create a new network from nodes and edges."""
        node_speeds: Dict[str, RandomVariable] = {}
        for n in nodes:
            if isinstance(n, StochasticNetworkNode):
                node_speeds[n.name] = n.speed
            elif isinstance(n, tuple) and len(n) == 2:
                node_speeds[n[0]] = (
                    n[1]
                    if isinstance(n[1], RandomVariable)
                    else RandomVariable(samples=[n[1]])
                )
            else:
                raise ValueError(f"Invalid node: {n}")

        edge_dict: Dict[Tuple[str, str], RandomVariable] = {}
        for e in edges:
            src, dst = (
                sorted((e.source, e.target))
                if isinstance(e, StochasticNetworkEdge)
                else sorted((e[0], e[1]))
            )
            default_speed: RandomVariable = (
                RandomVariable(samples=[0.0])
                if src != dst
                else RandomVariable(samples=[math.inf])
            )
            speed: RandomVariable = (
                e.speed
                if isinstance(e, StochasticNetworkEdge)
                else (
                    e[2]
                    if isinstance(e[2], RandomVariable)
                    else RandomVariable(samples=[e[2]])
                )
                if isinstance(e, tuple) and len(e) == 3
                else default_speed
            )
            if (src, dst) in edge_dict:
                continue
            edge_dict[(src, dst)] = speed

        for n1, n2 in product(node_speeds.keys(), repeat=2):
            src, dst = sorted((n1, n2))
            if (src, dst) not in edge_dict:
                default_speed = (
                    RandomVariable(samples=[0.0])
                    if src != dst
                    else RandomVariable(samples=[math.inf])
                )
                edge_dict[(src, dst)] = default_speed

        node_set: Set[StochasticNetworkNode] = {
            StochasticNetworkNode(name=name, speed=speed)
            for name, speed in node_speeds.items()
        }
        edge_set: Set[StochasticNetworkEdge] = {
            StochasticNetworkEdge(source=src, target=dst, speed=speed)
            for (src, dst), speed in edge_dict.items()
        }

        return cls(nodes=frozenset(node_set), edges=frozenset(edge_set))

    @cached_property
    def graph(self) -> nx.Graph:
        """Convert the network to a NetworkX graph."""
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node.name, weight=node.speed)
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, weight=edge.speed)
        return G

    @cached_property
    def _node_by_name(self) -> Dict[str, "StochasticNetworkNode"]:
        return {node.name: node for node in self.nodes}

    @cached_property
    def _edge_by_pair(self) -> Dict[Tuple[str, str], "StochasticNetworkEdge"]:
        pairs: Dict[Tuple[str, str], StochasticNetworkEdge] = {}
        for edge in self.edges:
            pairs[(edge.source, edge.target)] = edge
            if edge.source != edge.target:
                pairs[(edge.target, edge.source)] = StochasticNetworkEdge(
                    source=edge.target, target=edge.source, speed=edge.speed
                )
        return pairs

    @classmethod
    def from_nx(
        cls,
        G: nx.Graph,
        node_speed_attr: str = "weight",
        edge_speed_attr: str = "weight",
    ) -> "StochasticNetwork":
        """Create a StochasticNetwork from a NetworkX graph."""
        nodes = [
            StochasticNetworkNode(name=n, speed=G.nodes[n][node_speed_attr])
            for n in G.nodes
        ]
        edges = [
            StochasticNetworkEdge(
                source=u, target=v, speed=G.edges[u, v][edge_speed_attr]
            )
            for u, v in G.edges
        ]
        return StochasticNetwork.create(nodes=nodes, edges=edges)

    def scale_to_ccr(
        self,
        task_graph: "StochasticTaskGraph",
        target_ccr: float,
        free_speed_threshold: float = 1e9,
    ) -> "StochasticNetwork":
        """Scale link speeds to hit a target communication-to-computation ratio (CCR).

        The CCR is evaluated on means (what the scheduler plans on). Unlike the
        deterministic Network.scale_to_ccr, which sets every link to one uniform speed,
        this multiplies all finite inter-node links by a common factor so the tiered
        edge/fog/cloud bandwidth ratios are preserved. Self-loops and effectively
        infinite links (mean speed at or above free_speed_threshold, the shared-file-system
        cloud links) are left untouched.

        Args:
            task_graph (StochasticTaskGraph): The task graph to compute CCR against.
            target_ccr (float): The target CCR.
            free_speed_threshold (float): Links at or above this mean speed are treated as
                free and not scaled.

        Returns:
            StochasticNetwork: A new network with scaled links.
        """
        if target_ccr <= 0:
            raise ValueError("Target CCR must be positive.")

        def is_scalable(edge: StochasticNetworkEdge) -> bool:
            return edge.source != edge.target and edge.speed.mean() < free_speed_threshold

        avg_node_speed = sum(node.speed.mean() for node in self.nodes) / len(self.nodes)
        avg_task_cost = sum(task.cost.mean() for task in task_graph.tasks) / len(
            task_graph.tasks
        )
        avg_data_size = sum(dep.size.mean() for dep in task_graph.dependencies) / len(
            task_graph.dependencies
        )
        avg_comp_time = avg_task_cost / avg_node_speed
        target_link_speed = avg_data_size / (target_ccr * avg_comp_time)

        scalable = [edge for edge in self.edges if is_scalable(edge)]
        if not scalable:
            # No finite inter-node links (e.g. a single-node network): there is no
            # communication to scale, so CCR is undefined. Return the network unchanged.
            return self
        avg_link_speed = sum(edge.speed.mean() for edge in scalable) / len(scalable)
        alpha = target_link_speed / avg_link_speed

        scaled_edges = [
            StochasticNetworkEdge(
                source=edge.source, target=edge.target, speed=edge.speed * alpha
            )
            if is_scalable(edge)
            else edge
            for edge in self.edges
        ]
        return StochasticNetwork.create(nodes=self.nodes, edges=scaled_edges)

    def get_node(self, node: str | StochasticNetworkNode) -> StochasticNetworkNode:
        """Get a node by name."""
        name = node.name if isinstance(node, StochasticNetworkNode) else node
        result = self._node_by_name.get(name)
        if result is None:
            raise ValueError(f"Node {name} does not exist in the network.")
        return result

    def get_edge(
        self, source: str | StochasticNetworkNode, target: str | StochasticNetworkNode
    ) -> StochasticNetworkEdge:
        """Get the edge between two nodes."""
        source = source.name if isinstance(source, StochasticNetworkNode) else source
        target = target.name if isinstance(target, StochasticNetworkNode) else target
        edge = self._edge_by_pair.get((source, target))
        if edge is None:
            raise ValueError(f"Edge from {source} to {target} does not exist.")
        return edge


class StochasticTaskGraphNode(BaseModel):
    """A task in the task graph with stochastic cost."""

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    name: str = Field(..., description="The name of the task.")
    cost: RandomVariable = Field(
        ..., description="The cost (computation time) of the task."
    )

    def __hash__(self) -> int:
        return hash(self.name)

    def __lt__(self, other: "StochasticTaskGraphNode") -> bool:
        return self.name < other.name

    def sample(self) -> TaskGraphNode:
        """Sample the stochastic task to get a deterministic task."""
        return TaskGraphNode(name=self.name, cost=self.cost.sample(num_samples=1)[0])


class StochasticTaskGraphEdge(BaseModel):
    """A dependency edge in the task graph with stochastic size."""

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    source: str = Field(..., description="The source task of the edge.")
    target: str = Field(..., description="The target task of the edge.")
    size: RandomVariable = Field(..., description="The data size of the edge.")

    def __hash__(self) -> int:
        return hash((self.source, self.target))

    def __lt__(self, other: "StochasticTaskGraphEdge") -> bool:
        return (self.source, self.target) < (other.source, other.target)

    def sample(self) -> TaskGraphEdge:
        """Sample the stochastic edge to get a deterministic edge."""
        return TaskGraphEdge(
            source=self.source,
            target=self.target,
            size=self.size.sample(num_samples=1)[0],
        )


class StochasticTaskGraph(BaseModel):
    """A task graph of tasks and dependencies with stochastic values."""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    tasks: FrozenSet[StochasticTaskGraphNode] = Field(
        ..., description="The tasks in the task graph."
    )
    dependencies: FrozenSet[StochasticTaskGraphEdge] = Field(
        ..., description="The dependencies in the task graph."
    )

    @cached_property
    def computed_hash(self) -> int:
        return hash((frozenset(self.tasks), frozenset(self.dependencies)))

    def __hash__(self) -> int:
        return self.computed_hash

    def sample(self) -> TaskGraph:
        """Sample the stochastic task graph to get a deterministic task graph.

        Sorted iteration keeps the np.random draw order (and thus the realization)
        independent of PYTHONHASHSEED.
        """
        return TaskGraph.create(
            tasks=[task.sample() for task in sorted(self.tasks, key=lambda t: t.name)],
            dependencies=[
                dep.sample()
                for dep in sorted(self.dependencies, key=lambda d: (d.source, d.target))
            ],
        )

    @classmethod
    def create(
        cls,
        tasks: Iterable[StochasticTaskGraphNode | Tuple[str, RandomVariable | float]],
        dependencies: Iterable[
            StochasticTaskGraphEdge
            | Tuple[str, str, RandomVariable | float]
            | Tuple[str, str]
        ],
    ) -> "StochasticTaskGraph":
        """Create a new task graph from tasks and dependencies."""
        task_set: Set[StochasticTaskGraphNode] = set()
        for t in tasks:
            if isinstance(t, StochasticTaskGraphNode):
                task_set.add(t)
            elif isinstance(t, tuple) and len(t) == 2:
                task_cost = (
                    t[1]
                    if isinstance(t[1], RandomVariable)
                    else RandomVariable(samples=[t[1]])
                )
                task_set.add(StochasticTaskGraphNode(name=t[0], cost=task_cost))
            else:
                raise ValueError(f"Invalid task: {t}")
        dependency_set: Set[StochasticTaskGraphEdge] = set()
        for d in dependencies:
            if isinstance(d, StochasticTaskGraphEdge):
                dependency_set.add(d)
            elif isinstance(d, tuple) and len(d) == 3:
                size = (
                    d[2]
                    if isinstance(d[2], RandomVariable)
                    else RandomVariable(samples=[d[2]])
                )
                dependency_set.add(
                    StochasticTaskGraphEdge(source=d[0], target=d[1], size=size)
                )
            elif isinstance(d, tuple) and len(d) == 2:
                dependency_set.add(
                    StochasticTaskGraphEdge(
                        source=d[0], target=d[1], size=RandomVariable(samples=[0.0])
                    )
                )
            else:
                raise ValueError(f"Invalid dependency: {d}")

        # ensure there is one source and one sink
        sources = [
            t for t in task_set if all(d.target != t.name for d in dependency_set)
        ]
        sinks = [t for t in task_set if all(d.source != t.name for d in dependency_set)]
        if len(sources) != 1:
            logging.warning("Task graph has multiple sources; adding super source.")
            super_source = StochasticTaskGraphNode(
                name="__super_source__", cost=RandomVariable(samples=[0.0])
            )
            task_set.add(super_source)
            for source in sources:
                dependency_set.add(
                    StochasticTaskGraphEdge(
                        source=super_source.name,
                        target=source.name,
                        size=RandomVariable(samples=[0.0]),
                    )
                )
        if len(sinks) != 1:
            logging.warning("Task graph has multiple sinks; adding super sink.")
            super_sink = StochasticTaskGraphNode(
                name="__super_sink__", cost=RandomVariable(samples=[0.0])
            )
            task_set.add(super_sink)
            for sink in sinks:
                dependency_set.add(
                    StochasticTaskGraphEdge(
                        source=sink.name,
                        target=super_sink.name,
                        size=RandomVariable(samples=[0.0]),
                    )
                )

        return cls(
            tasks=frozenset(task_set), dependencies=frozenset(dependency_set)
        )

    @cached_property
    def graph(self) -> nx.DiGraph:
        """Convert the task graph to a NetworkX directed graph."""
        G = nx.DiGraph()
        for task in self.tasks:
            G.add_node(task.name, weight=task.cost)
        for dependency in self.dependencies:
            G.add_edge(dependency.source, dependency.target, weight=dependency.size)
        return G

    @cached_property
    def _task_by_name(self) -> Dict[str, "StochasticTaskGraphNode"]:
        return {task.name: task for task in self.tasks}

    @cached_property
    def _in_edges_by_task(self) -> Dict[str, List["StochasticTaskGraphEdge"]]:
        result: Dict[str, List[StochasticTaskGraphEdge]] = {t.name: [] for t in self.tasks}
        # Sorted so edge order (and any tie-breaking on it) is PYTHONHASHSEED-independent.
        for dep in sorted(self.dependencies, key=lambda d: (d.source, d.target)):
            result.setdefault(dep.target, []).append(dep)
        return result

    @cached_property
    def _out_edges_by_task(self) -> Dict[str, List["StochasticTaskGraphEdge"]]:
        result: Dict[str, List[StochasticTaskGraphEdge]] = {t.name: [] for t in self.tasks}
        for dep in sorted(self.dependencies, key=lambda d: (d.source, d.target)):
            result.setdefault(dep.source, []).append(dep)
        return result

    @cached_property
    def _dependency_by_pair(self) -> Dict[Tuple[str, str], "StochasticTaskGraphEdge"]:
        return {(dep.source, dep.target): dep for dep in self.dependencies}

    @classmethod
    def from_nx(
        cls,
        G: nx.DiGraph,
        node_weight_attr: str = "weight",
        edge_weight_attr: str = "weight",
    ) -> "StochasticTaskGraph":
        """Create a StochasticTaskGraph from a NetworkX directed graph."""
        tasks = [
            StochasticTaskGraphNode(name=n, cost=G.nodes[n][node_weight_attr])
            for n in G.nodes
        ]
        dependencies = [
            StochasticTaskGraphEdge(
                source=u, target=v, size=G.edges[u, v][edge_weight_attr]
            )
            for u, v in G.edges
        ]
        return StochasticTaskGraph.create(tasks=tasks, dependencies=dependencies)

    def topological_sort(self) -> List[StochasticTaskGraphNode]:
        """Get a topological sort of the task graph."""
        sorted_task_names = list(nx.topological_sort(self.graph))
        name_to_task = {task.name: task for task in self.tasks}
        return [name_to_task[name] for name in sorted_task_names]

    def all_topological_sorts(
        self,
    ) -> Generator[List[StochasticTaskGraphNode], None, None]:
        """Generate all topological sorts of the task graph."""
        import networkx.algorithms.dag as dag

        name_to_task = {task.name: task for task in self.tasks}
        for sort in dag.all_topological_sorts(self.graph):
            yield [name_to_task[name] for name in sort]

    def in_degree(self, task: str | StochasticTaskGraphNode) -> int:
        """Get the in-degree of a task."""
        task = task.name if isinstance(task, StochasticTaskGraphNode) else task
        return self.graph.in_degree(task)

    def in_edges(
        self, task: str | StochasticTaskGraphNode
    ) -> List[StochasticTaskGraphEdge]:
        """Get the incoming edges of a task."""
        task = task.name if isinstance(task, StochasticTaskGraphNode) else task
        return self._in_edges_by_task.get(task, [])

    def out_degree(self, task: str | StochasticTaskGraphNode) -> int:
        """Get the out-degree of a task."""
        task = task.name if isinstance(task, StochasticTaskGraphNode) else task
        return self.graph.out_degree(task)

    def out_edges(
        self, task: str | StochasticTaskGraphNode
    ) -> List[StochasticTaskGraphEdge]:
        """Get the outgoing edges of a task."""
        task = task.name if isinstance(task, StochasticTaskGraphNode) else task
        return self._out_edges_by_task.get(task, [])

    def get_task(self, name: str | StochasticTaskGraphNode) -> StochasticTaskGraphNode:
        """Get a task by name."""
        name = name.name if isinstance(name, StochasticTaskGraphNode) else name
        task = self._task_by_name.get(name)
        if task is None:
            raise ValueError(f"Task {name} does not exist in the task graph.")
        return task

    def get_dependency(
        self,
        source: str | StochasticTaskGraphNode,
        target: str | StochasticTaskGraphNode,
    ) -> StochasticTaskGraphEdge:
        """Get the dependency edge between two tasks."""
        source = source.name if isinstance(source, StochasticTaskGraphNode) else source
        target = target.name if isinstance(target, StochasticTaskGraphNode) else target
        edge = self._dependency_by_pair.get((source, target))
        if edge is None:
            raise ValueError(f"Dependency from {source} to {target} does not exist.")
        return edge


class StochasticScheduledTask(BaseModel):
    """A scheduled task with stochastic timing."""

    model_config = {"arbitrary_types_allowed": True}

    node: str = Field(..., description="The node the task is scheduled on.")
    name: str = Field(..., description="The name of the task.")
    rank: float = Field(
        ...,
        description="The rank of the task, determining the order of execution on the node.",
    )

    def __str__(self) -> str:
        return f"Task(node={self.node}, name={self.name}, rank={self.rank})"


class StochasticSchedule(BaseModel):
    """A schedule of tasks on nodes with stochastic timing."""

    model_config = {"arbitrary_types_allowed": True}

    task_graph: StochasticTaskGraph = Field(
        ..., description="The task graph being scheduled."
    )
    network: StochasticNetwork = Field(
        ..., description="The network the tasks are scheduled on."
    )
    mapping: Dict[str, List[StochasticScheduledTask]] = Field(
        ..., description="The mapping of tasks to nodes."
    )
    node_constraints: Optional[Dict[str, Set[str]]] = Field(
        default=None,
        description=(
            "Optional per-task placement constraints for this instance, mapping a task "
            "name to the set of node names it may run on. Tasks absent from the map may "
            "run on any node. None (the default) means no constraints."
        ),
    )

    _task_map: Dict[str, StochasticScheduledTask] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        task_graph: StochasticTaskGraph,
        network: StochasticNetwork,
        mapping: Optional[Dict[str, List[StochasticScheduledTask]]] = None,
        node_constraints: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        super().__init__(
            task_graph=task_graph,
            network=network,
            mapping=(
                mapping
                if mapping is not None
                else {node.name: [] for node in network.nodes}
            ),
            node_constraints=node_constraints,
        )

    def allowed_nodes(self, task_name: str) -> Optional[Set[str]]:
        """Return the node names `task_name` may run on, or None if it is unconstrained."""
        if self.node_constraints is None:
            return None
        return self.node_constraints.get(task_name)

    @property
    def makespan(self) -> RandomVariable:
        """Get the makespan of the schedule."""
        return RandomVariable(
            samples=[self.sample().makespan for _ in range(DEFAULT_NUM_SAMPLES)]
        )

    def sample(self) -> Schedule:
        """Sample the stochastic schedule to get a deterministic schedule."""
        return self.determinize(
            network=self.network.sample(),
            task_graph=self.task_graph.sample()
        )
    
    def determinize(self, network: Network, task_graph: TaskGraph) -> Schedule:
        schedule = Schedule(
            task_graph=task_graph, network=network, node_constraints=self.node_constraints
        )
        # The middle int is a monotonic tiebreaker so equal ranks never fall through
        # to comparing StochasticScheduledTask objects (which are not orderable).
        pq: PriorityQueue[tuple[float, int, StochasticScheduledTask]] = PriorityQueue()
        tiebreak = 0
        task_by_name = {
            task.name: task for tasks in self.mapping.values() for task in tasks
        }
        # add all source tasks to the priority queue
        for _, tasks in self.mapping.items():
            for task in tasks:
                if task_graph.in_degree(task.name) == 0:
                    pq.put((-task.rank, tiebreak, task))
                    tiebreak += 1

        while not pq.empty():
            _, _, task = pq.get()

            det_task = task_graph.get_task(task.name)
            det_node = network.get_node(task.node)
            start_time = schedule.get_earliest_start_time(
                det_task, det_node, append_only=True
            )
            schedule.add_task(
                task=ScheduledTask(
                    node=task.node,
                    name=task.name,
                    start=start_time,
                    end=start_time + (det_task.cost / det_node.speed),
                )
            )

            # add dependent tasks to the priority queue
            # if all of their dependencies are scheduled
            for out_edge in task_graph.out_edges(task.name):
                if all(
                    schedule.is_scheduled(in_edge.source)
                    for in_edge in task_graph.in_edges(out_edge.target)
                ):
                    succ_task = task_by_name.get(out_edge.target)
                    if succ_task is None:
                        raise ValueError(f"Could not find successor task {out_edge.target!r}")
                    pq.put((-succ_task.rank, tiebreak, succ_task))
                    tiebreak += 1
        return schedule

    def __getitem__(
        self, node: str | StochasticNetworkNode
    ) -> List[StochasticScheduledTask]:
        """Get the tasks scheduled on a node."""
        node = node.name if isinstance(node, StochasticNetworkNode) else node
        if node not in self.mapping:
            raise ValueError(
                f"Node {node} not in schedule. Nodes are {set(self.mapping.keys())}."
            )
        return self.mapping[node]

    def items(self) -> Iterable[Tuple[str, List[StochasticScheduledTask]]]:
        """Get the items of the schedule."""
        return self.mapping.items()

    def add_task(self, task: StochasticScheduledTask) -> None:
        """Add a task to the schedule."""
        if task.node not in self.mapping:
            raise ValueError(
                f"Node {task.node} not in schedule. Nodes are {set(self.mapping.keys())}."
            )
        allowed = self.allowed_nodes(task.name)
        if allowed is not None and task.node not in allowed:
            raise ConstraintViolation(
                f"Task {task.name} placed on node {task.node}, but its constraint only "
                f"allows {sorted(allowed)}."
            )
        idx = bisect.bisect_left([t.rank for t in self.mapping[task.node]], task.rank)
        self.mapping[task.node].insert(idx, task)
        self._task_map[task.name] = task

    def remove_task(self, task_name: str | StochasticTaskGraphNode) -> None:
        """Remove a task from the schedule."""
        task_name = (
            task_name.name
            if isinstance(task_name, StochasticTaskGraphNode)
            else task_name
        )
        if task_name not in self._task_map:
            raise ValueError(f"Task {task_name} not in schedule.")
        scheduled_task = self._task_map[task_name]
        self.mapping[scheduled_task.node].remove(scheduled_task)
        del self._task_map[task_name]

    def is_scheduled(self, task_name: str) -> bool:
        """Check if a task is scheduled."""
        return task_name in self._task_map

    def get_scheduled_task(self, task_name: str) -> StochasticScheduledTask:
        """Get the scheduled task by name."""
        if task_name not in self._task_map:
            raise ValueError(f"Task {task_name} not in schedule.")
        return self._task_map[task_name]


class StochasticScheduler(ABC):
    """An abstract class for a stochastic scheduler."""

    @abstractmethod
    def schedule(
        self, network: StochasticNetwork, task_graph: StochasticTaskGraph
    ) -> StochasticSchedule:
        """Schedule the tasks on the network."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Get the name of the scheduler."""
        return self.__class__.__name__
    
    
