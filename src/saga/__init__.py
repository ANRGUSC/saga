from abc import ABC, abstractmethod
import logging
from typing import (
    Any, Dict, FrozenSet, Generator, Iterable, List,
    Optional, Set, Tuple, Type, TypeVar, Union,
    Generic, cast
)
from pydantic import BaseModel, PrivateAttr, Field
from functools import cached_property
import math
import networkx as nx
import bisect
from itertools import product

from saga.utils.random_variable import RandomVariable, rv_max

Deterministic = float
Stochastic = float | RandomVariable
NumericT = TypeVar("NumericT", bound=Union[Deterministic, Stochastic], default=Deterministic)

class NetworkNode(BaseModel, Generic[NumericT]): 
    """A node in the network."""
    model_config = {'frozen': False}

    name: str = Field(..., description="The name of the node.")
    speed: NumericT = Field(..., description="The speed of the node.")

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __lt__(self, other: "NetworkNode") -> bool:
        return self.name < other.name

class NetworkEdge(BaseModel, Generic[NumericT]):
    """An edge in the network."""
    model_config = {'frozen': False}

    source: str = Field(..., description="The source node of the edge.")
    target: str = Field(..., description="The target node of the edge.")
    speed: NumericT = Field(..., description="The speed (bandwidth) of the edge.")

    def __hash__(self) -> int:
        return hash((self.source, self.target))

    def model_post_init(self, __context: Any) -> None:
        """Set default speed based on source and target if not explicitly provided."""
        if self.speed == 0.0 and self.source == self.target:
            object.__setattr__(self, 'speed', math.inf)

    def __lt__(self, other: "NetworkEdge") -> bool:
        return (self.source, self.target) < (other.source, other.target)

class Network(BaseModel, Generic[NumericT]):
    """A network of nodes and edges."""
    model_config = {'frozen': True, 'arbitrary_types_allowed': False}

    nodes: FrozenSet[NetworkNode[NumericT]] = Field(..., description="The nodes in the network.")
    edges: FrozenSet[NetworkEdge[NumericT]] = Field(..., description="The edges in the network.")

    @classmethod
    def create(cls,
               nodes: Iterable[NetworkNode | Tuple[str, NumericT]],
               edges: Iterable[NetworkEdge | Tuple[str, str, NumericT] | Tuple[str, str]]) -> 'Network[NumericT]':
        """Create a new network from nodes and edges.

        Args:
            nodes: An iterable of NetworkNode objects or tuples (name, speed).
            edges: An iterable of NetworkEdge objects or tuples (source, target) or (source, target, speed).

        Returns:
            Network: A new network instance.
        """
        node_speeds = {}
        for n in nodes:
            if isinstance(n, NetworkNode):
                node_speeds[n.name] = n.speed
            elif isinstance(n, tuple) and len(n) == 2:
                node_speeds[n[0]] = n[1]
            else:
                raise ValueError(f"Invalid node: {n}")
            
        edge_dict = {}
        for e in edges:
            src, dst = sorted((e.source, e.target)) if isinstance(e, NetworkEdge) else sorted((e[0], e[1]))
            default_speed = 0.0 if src != dst else math.inf
            speed = (
                e.speed if isinstance(e, NetworkEdge) else
                e[2] if isinstance(e, tuple) and len(e) == 3 else
                default_speed
            )
            if (src, dst) in edge_dict:
                if edge_dict[(src, dst)] != speed:
                    raise ValueError(f"Duplicate edge between {src} and {dst}")
                continue
            edge_dict[(src, dst)] = speed

        for n1, n2 in product(node_speeds.keys(), repeat=2):
            src, dst = sorted((n1, n2))
            if (src, dst) not in edge_dict:
                default_speed = 0.0 if src != dst else math.inf
                edge_dict[(src, dst)] = default_speed

        node_set: Set[NetworkNode[NumericT]] = {
            NetworkNode(name=name, speed=speed)
            for name, speed in node_speeds.items()
        }
        edge_set: Set[NetworkEdge[NumericT]] = {
            NetworkEdge(source=src, target=dst, speed=speed)
            for (src, dst), speed in edge_dict.items()
        }

        return cls(
            nodes=frozenset(node_set),
            edges=frozenset(edge_set)
        )

    def scale_to_ccr(self, task_graph: "TaskGraph", target_ccr: float) -> "Network[NumericT]":
        """Scale the network to achieve a target communication-to-computation ratio (CCR) with respect to a task graph.

        Args:
            task_graph (TaskGraph): The task graph to compute CCR against.
            target_ccr (float): The target CCR.

        Returns:
            Network: A new scaled network.
        """
        if target_ccr <= 0:
            raise ValueError("Target CCR must be positive.")
        avg_node_speed = sum(node.speed for node in self.nodes) / len(self.nodes)
        avg_task_cost = sum(task.cost for task in task_graph.tasks) / len(task_graph.tasks)
        avg_data_size = sum(dep.size for dep in task_graph.dependencies) / len(task_graph.dependencies)
        avg_comp_time = avg_task_cost / avg_node_speed
        link_speed = cast(NumericT, avg_data_size / (target_ccr * avg_comp_time))
        scaled_edges: Set[NetworkEdge[NumericT]] = set()
        for edge in self.edges:
            scaled_edges.add(NetworkEdge[NumericT](
                source=edge.source,
                target=edge.target,
                speed=link_speed
            ))
        network = Network(nodes=self.nodes, edges=frozenset(scaled_edges))
        return network
            
    
    @cached_property
    def graph(self) -> nx.Graph:
        """Convert the network to a NetworkX graph.

        Returns:
            nx.Graph: The NetworkX graph representation of the network.
        """
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node.name, weight=node.speed)
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, weight=edge.speed)
        return G
    
    @classmethod
    def from_nx(cls,
                G: nx.Graph,
                node_speed_attr: str = "weight",
                edge_speed_attr: str = "weight") -> 'Network':
        """Create a Network from a NetworkX graph.

        Args:
            G (nx.Graph): The NetworkX graph.
            node_speed_attr (str, optional): The attribute name for the speed of nodes. Defaults to "speed".
            edge_speed_attr (str, optional): The attribute name for the speed of edges. Defaults to "speed".

        Returns:
            Network: The Network representation of the graph.
        """
        nodes = [
            NetworkNode(name=n, speed=G.nodes[n][node_speed_attr])
            for n in G.nodes
        ]
        edges = [
            NetworkEdge(source=u, target=v, speed=G.edges[u, v][edge_speed_attr])
            for u, v in G.edges
        ]
        return Network.create(nodes=nodes, edges=edges)

    def get_node(self, node: str | NetworkNode[NumericT]) -> NetworkNode[NumericT]:
        """Get a node

        Args:
            node (str | NetworkNode): The node or the name of the node.

        Returns:
            NetworkNode: The node with the given name.

        Raises:
            ValueError: If the node does not exist.
        """
        name = node.name if isinstance(node, NetworkNode) else node
        for _node in self.nodes:
            if _node.name == name:
                return _node
        raise ValueError(f"Node {name} does not exist in the network.")

    def get_edge(self,
                 source: str | NetworkNode[NumericT],
                 target: str | NetworkNode[NumericT]) -> NetworkEdge[NumericT]:
        """Get the edge between two nodes.

        Args:
            source (str): The source node.
            target (str): The target node.

        Returns:
            NetworkEdge: The edge between the source and target nodes.

        Raises:
            ValueError: If the edge does not exist.
        """
        source = source.name if isinstance(source, NetworkNode) else source
        target = target.name if isinstance(target, NetworkNode) else target
        edge = self.graph.get_edge_data(source, target)
        if edge is None:
            raise ValueError(f"Edge from {source} to {target} does not exist.")
        return NetworkEdge(source=source, target=target, speed=edge['weight'])

class TaskGraphNode(BaseModel, Generic[NumericT]):
    """A task in the task graph."""
    model_config = {'frozen': False}

    name: str = Field(..., description="The name of the task.")
    cost: NumericT = Field(..., description="The cost (computation time) of the task.")

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __lt__(self, other: "TaskGraphNode[NumericT]") -> bool:
        return self.name < other.name
    
class TaskGraphEdge(BaseModel, Generic[NumericT]):
    """A dependency edge in the task graph."""
    model_config = {'frozen': False}

    source: str = Field(..., description="The source task of the edge.")
    target: str = Field(..., description="The target task of the edge.")
    size: NumericT = Field(..., description="The data size of the edge.")

    def __hash__(self) -> int:
        return hash((self.source, self.target))
    
    def __lt__(self, other: "TaskGraphEdge[NumericT]") -> bool:
        return (self.source, self.target) < (other.source, other.target)
    
class TaskGraph(BaseModel, Generic[NumericT]):
    """A task graph of tasks and dependencies."""
    model_config = {'frozen': True, 'arbitrary_types_allowed': False}

    tasks: FrozenSet[TaskGraphNode[NumericT]] = Field(..., description="The tasks in the task graph.")
    dependencies: FrozenSet[TaskGraphEdge[NumericT]] = Field(..., description="The dependencies in the task graph.")

    @classmethod
    def create(cls,
               tasks: Iterable[TaskGraphNode | Tuple[str, NumericT]],
               dependencies: Iterable[TaskGraphEdge | Tuple[str, str, NumericT] | Tuple[str, str]]) -> 'TaskGraph[NumericT]':
        """Create a new task graph from tasks and dependencies.

        Args:
            tasks: An iterable of TaskGraphNode objects or tuples (name, weight).
            dependencies: An iterable of TaskGraphEdge objects or tuples (source, target) or (source, target, weight).

        Returns:
            TaskGraph: A new task graph instance.
        """
        task_set = set()
        for t in tasks:
            if isinstance(t, TaskGraphNode):
                task_set.add(t)
            elif isinstance(t, tuple) and len(t) == 2:
                task_set.add(TaskGraphNode(name=t[0], cost=t[1]))
            else:
                raise ValueError(f"Invalid task: {t}")
        dependency_set = set()
        for d in dependencies:
            if isinstance(d, TaskGraphEdge):
                dependency_set.add(d)
            elif isinstance(d, tuple) and len(d) == 3:
                dependency_set.add(TaskGraphEdge(source=d[0], target=d[1], size=d[2]))
            elif isinstance(d, tuple) and len(d) == 2:
                dependency_set.add(TaskGraphEdge(source=d[0], target=d[1], size=0.0))
            else:
                raise ValueError(f"Invalid dependency: {d}")

        # ensure there is one source and one sink
        sources = [t for t in task_set if all(d.target != t.name for d in dependency_set)]
        sinks = [t for t in task_set if all(d.source != t.name for d in dependency_set)]
        if len(sources) != 1:
            logging.warning("Task graph has multiple sources; adding super source.")
            super_source = TaskGraphNode(name="__super_source__", cost=0.0)
            task_set.add(super_source)
            for source in sources:
                dependency_set.add(TaskGraphEdge(source=super_source.name, target=source.name, size=0.0))
        if len(sinks) != 1:
            logging.warning("Task graph has multiple sinks; adding super sink.")
            super_sink = TaskGraphNode(name="__super_sink__", cost=0.0)
            task_set.add(super_sink)
            for sink in sinks:
                dependency_set.add(TaskGraphEdge(source=sink.name, target=super_sink.name, size=0.0))
        
        return cls(
            tasks=frozenset(task_set),
            dependencies=frozenset(dependency_set)
        )

    @cached_property
    def graph(self) -> nx.DiGraph:
        """Convert the task graph to a NetworkX directed graph.

        Returns:
            nx.DiGraph: The NetworkX directed graph representation of the task graph.
        """
        G = nx.DiGraph()
        for task in self.tasks:
            G.add_node(task.name, weight=task.cost)
        for dependency in self.dependencies:
            G.add_edge(dependency.source, dependency.target, weight=dependency.size)
        return G
    
    @classmethod
    def from_nx(cls, G: nx.DiGraph, node_weight_attr: str = "weight", edge_weight_attr: str = "weight") -> 'TaskGraph':
        """Create a TaskGraph from a NetworkX directed graph.

        Args:
            G (nx.DiGraph): The NetworkX directed graph.
            node_weight_attr (str, optional): The attribute name for the weight of nodes. Defaults to "weight".
            edge_weight_attr (str, optional): The attribute name for the weight of edges. Defaults to "weight".

        Returns:
            TaskGraph: The TaskGraph representation of the directed graph.
        """
        tasks = [TaskGraphNode(name=n, cost=G.nodes[n][node_weight_attr]) for n in G.nodes]
        dependencies = [TaskGraphEdge(source=u, target=v, size=G.edges[u, v][edge_weight_attr]) for u, v in G.edges]
        return TaskGraph.create(tasks=tasks, dependencies=dependencies)
    
    def topological_sort(self) -> List[TaskGraphNode[NumericT]]:
        """Get a topological sort of the task graph.

        Returns:
            List[TaskGraphNode]: A list of tasks in topological order.
        """
        sorted_task_names = list(nx.topological_sort(self.graph))
        name_to_task = {task.name: task for task in self.tasks}
        return [name_to_task[name] for name in sorted_task_names]
    
    def all_topological_sorts(self) -> Generator[List[TaskGraphNode[NumericT]], None, None]:
        """Generate all topological sorts of the task graph.

        Yields:
            Generator[List[TaskGraphNode], None, None]: A generator of lists of tasks in topological order.
        """
        import networkx.algorithms.dag as dag
        name_to_task = {task.name: task for task in self.tasks}
        for sort in dag.all_topological_sorts(self.graph):
            yield [name_to_task[name] for name in sort]
    
    def in_degree(self, task: str | TaskGraphNode) -> int:
        """Get the in-degree of a task.

        Args:
            task (str | TaskGraphNode): The task or the name of the task.
        Returns:
            int: The in-degree of the task.
        """
        task = task.name if isinstance(task, TaskGraphNode) else task
        return self.graph.in_degree(task)
    
    def in_edges(self, task: str | TaskGraphNode[NumericT]) -> List[TaskGraphEdge[NumericT]]:
        """Get the incoming edges of a task.

        Args:
            task (str | TaskGraphNode): The task or the name of the task.
        Returns:
            List[TaskGraphEdge]: A list of incoming edges to the task.
        """
        task = task.name if isinstance(task, TaskGraphNode) else task
        edges = []
        for src, tgt in self.graph.in_edges(task):
            edge_data = self.graph.get_edge_data(src, tgt)
            edges.append(TaskGraphEdge(source=src, target=tgt, size=edge_data['weight']))
        return edges
    
    def out_degree(self, task: str | TaskGraphNode) -> int:
        """Get the out-degree of a task.

        Args:
            task (str | TaskGraphNode): The task or the name of the task.
        Returns:
            int: The out-degree of the task.
        """
        task = task.name if isinstance(task, TaskGraphNode) else task
        out_degree = self.graph.out_degree(task)
        return out_degree
    
    def out_edges(self, task: str | TaskGraphNode) -> List[TaskGraphEdge]:
        """Get the outgoing edges of a task.

        Args:
            task (str | TaskGraphNode): The task or the name of the task.
        Returns:
            List[TaskGraphEdge]: A list of outgoing edges from the task.
        """
        task = task.name if isinstance(task, TaskGraphNode) else task
        edges = []
        for src, tgt in self.graph.out_edges(task):
            edge_data = self.graph.get_edge_data(src, tgt)
            edges.append(TaskGraphEdge(source=src, target=tgt, size=edge_data['weight']))
        return edges
    
    def get_task(self, name: str | TaskGraphNode[NumericT]) -> TaskGraphNode[NumericT]:
        """Get a task by name.

        Args:
            name (str | TaskGraphNode): The task or the name of the task.
        Returns:
            TaskGraphNode: The task with the given name.

        Raises:
            ValueError: If the task does not exist.
        """
        name = name.name if isinstance(name, TaskGraphNode) else name
        for task in self.tasks:
            if task.name == name:
                return task
        raise ValueError(f"Task {name} does not exist in the task graph.")
    
    def get_dependency(self, source: str | TaskGraphNode, target: str | TaskGraphNode) -> TaskGraphEdge:
        """Get the dependency edge between two tasks.

        Args:
            source (str | TaskGraphNode): The source task.
            target (str | TaskGraphNode): The target task.
        Returns:
            TaskGraphEdge: The dependency edge between the source and target tasks.

        Raises:
            ValueError: If the dependency does not exist.
        """
        source = source.name if isinstance(source, TaskGraphNode) else source
        target = target.name if isinstance(target, TaskGraphNode) else target
        edge = self.graph.get_edge_data(source, target)
        if edge is None:
            raise ValueError(f"Dependency from {source} to {target} does not exist.")
        return TaskGraphEdge(source=source, target=target, size=edge['weight'])


class ScheduledTask(BaseModel, Generic[NumericT]):
    """A scheduled task."""
    node: str = Field(..., description="The node the task is scheduled on.")
    name: str = Field(..., description="The name of the task.")
    start: NumericT = Field(..., description="The start time of the task.")
    end: NumericT = Field(..., description="The end time of the task.")

    def __str__(self) -> str:
        return f"Task(node={self.node}, name={self.name}, start={self.start:0.2f}, end={self.end:0.2f})"

class Schedule(BaseModel, Generic[NumericT]):
    """A schedule of tasks on nodes."""

    task_graph: "TaskGraph[NumericT]" = Field(..., description="The task graph being scheduled.")
    network: "Network[NumericT]" = Field(..., description="The network the tasks are scheduled on.")
    mapping: Dict[str, List[ScheduledTask[NumericT]]] = Field(..., description="The mapping of tasks to nodes.")

    _task_map: Dict[str, ScheduledTask[NumericT]] = PrivateAttr(default_factory=dict)

    def __init__(self,
                 task_graph: "TaskGraph",
                 network: "Network",
                 mapping: Optional[Dict[str, List[ScheduledTask[NumericT]]]] = None) -> None:
        super().__init__(
            task_graph=task_graph,
            network=network,
            mapping=(
                mapping if mapping is not None else
                {node.name: [] for node in network.nodes}
            )
        )

    @property
    def makespan(self) -> NumericT:
        """Get the makespan of the schedule.

        Returns:
            float: The makespan of the schedule.
        """
        if not any(self.mapping.values()):
            return cast(NumericT, 0.0)
        ends = (tasks[-1].end for tasks in self.mapping.values() if tasks)
        return cast(NumericT, rv_max(ends))

    def __getitem__(self, node: str | NetworkNode[NumericT]) -> List[ScheduledTask[NumericT]]:
        """Get the tasks scheduled on a node.

        Args:
            node (str): The node to get the tasks for.

        Returns:
            List[Task]: A list of tasks scheduled on the node.
        """
        node = node.name if isinstance(node, NetworkNode) else node
        if node not in self.mapping:
            raise ValueError(f"Node {node} not in schedule. Nodes are {set(self.mapping.keys())}.")
        return self.mapping[node]
    
    def items(self) -> Iterable[Tuple[str, List[ScheduledTask[NumericT]]]]:
        """Get the items of the schedule.

        Returns:
            Iterable[Tuple[str, List[ScheduledTask]]]: An iterable of (node, tasks) pairs.
        """
        return self.mapping.items()

    def get_earliest_start_time(self,
                                task: str | TaskGraphNode[NumericT],
                                node: str | NetworkNode[NumericT],
                                append_only: bool = False) -> NumericT:
        """Get the earliest start time for a task on a node given the minimum start time and execution time.
        
        Args:
            task (str | TaskGraphNode): The task or the name of the task.
            node (str | NetworkNode): The node or the name of the node.
            append_only (bool): If True, only consider appending the task at the end of the schedule.
            
        Returns:
            float: The earliest start time of the task.
        """
        node = self.network.get_node(node)
        task = self.task_graph.get_task(task)

        exec_time = task.cost / node.speed
        min_start_time = 0.0
        for dependency in self.task_graph.in_edges(task):
            if not self.is_scheduled(dependency.source):
                raise ValueError(f"Parent task {dependency.source} is not scheduled yet.")
            parent_task = self._task_map[dependency.source]
            network_edge = self.network.get_edge(parent_task.node, node.name)
            arrival_time = parent_task.end + (dependency.size / network_edge.speed)
            min_start_time = rv_max([min_start_time, arrival_time])
    
        if append_only:
            min_start_time = (
                rv_max([min_start_time, self.mapping[node.name][-1].end])
                if self.mapping[node.name] else min_start_time
            )
            return cast(NumericT, min_start_time)
        else:
            if not self.mapping[node.name] or min_start_time + exec_time <= self.mapping[node.name][0].start:
                return cast(NumericT, min_start_time)
            for left, right in zip(self.mapping[node.name], self.mapping[node.name][1:]):
                if min_start_time >= left.end and min_start_time + exec_time <= right.start:
                    return cast(NumericT, min_start_time)
                elif min_start_time < left.end and left.end + exec_time <= right.start:
                    return left.end
            min_start_time = rv_max([min_start_time, self.mapping[node.name][-1].end])
            return cast(NumericT, min_start_time)

    def add_task(self, task: ScheduledTask[NumericT]) -> None:
        """Add a task to the schedule.

        Args:
            task (Task): The task to add.
        """
        if task.node not in self.mapping:
            raise ValueError(f"Node {task.node} not in schedule. Nodes are {set(self.mapping.keys())}.")
        # Use (start, end) tuple for comparison so tasks with same start time are ordered by end time
        idx = bisect.bisect_left([(t.start, t.end) for t in self.mapping[task.node]], (task.start, task.end))
        self.mapping[task.node].insert(idx, task)
        # check that next task starts after this task ends
        if idx + 1 < len(self.mapping[task.node]) and self.mapping[task.node][idx + 1].start < task.end:
            raise ValueError(f"Task {task} overlaps with next task {self.mapping[task.node][idx + 1]}.")

        self._task_map[task.name] = task

    def remove_task(self, task_name: str | TaskGraphNode[NumericT]) -> None:
        """Remove a task from the schedule.

        Args:
            task_name (str | TaskGraphNode): The task or the name of the task to remove.
        """
        task_name = task_name.name if isinstance(task_name, TaskGraphNode) else task_name
        if task_name not in self._task_map:
            raise ValueError(f"Task {task_name} not in schedule.")
        scheduled_task = self._task_map[task_name]
        self.mapping[scheduled_task.node].remove(scheduled_task)
        del self._task_map[task_name]

    def is_scheduled(self, task_name: str) -> bool:
        """Check if a task is scheduled.

        Args:
            task_name (str): The name of the task. 
        Returns:
            bool: True if the task is scheduled, False otherwise.
        """
        for tasks in self.mapping.values():
            if any(task.name == task_name for task in tasks):
                return True
        return False


class Scheduler(ABC):
    """An abstract class for a scheduler."""

    @abstractmethod
    def schedule(self, network: Network, task_graph: TaskGraph) -> Schedule:
        """Schedule the tasks on the network.

        Args:
            network (Network): The network to schedule on.
            task_graph (TaskGraph): The task graph to schedule.

        Returns:
            Schedule: The resulting schedule.
        """
        raise NotImplementedError
    
    def get_loaded_schedulers(self) -> List[Type['Scheduler']]:
        """Get the list of loaded schedulers.

        Returns:
            List[Type['Scheduler']]: A list of loaded schedulers.
        """
        return Scheduler.__subclasses__()
