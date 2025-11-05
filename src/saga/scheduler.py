from abc import ABC, abstractmethod
from functools import lru_cache, cached_property
import math
from typing import Any, Dict, FrozenSet, Generator, Iterable, List, Optional, Set, Tuple, Type
import networkx as nx
from pydantic import BaseModel, RootModel, computed_field, Field
import bisect
from itertools import combinations

class ScheduledTask(BaseModel):
    """A scheduled task."""
    node: str = Field(..., description="The node the task is scheduled on.")
    name: str = Field(..., description="The name of the task.")
    start: float = Field(..., description="The start time of the task.")
    end: float = Field(..., description="The end time of the task.")

    def __str__(self) -> str:
        return f"Task(node={self.node}, name={self.name}, start={self.start:0.2f}, end={self.end:0.2f})"

class Schedule(RootModel[Dict[str, List[ScheduledTask]]], ABC):
    """A schedule of tasks on nodes."""

    root: Dict[str, List[ScheduledTask]] = Field(
        default_factory=dict,
        description="The mapping of nodes to their scheduled tasks."
    )

    @classmethod
    def create(cls, nodes: Iterable[str]) -> 'Schedule':
        """Create a new schedule.

        Args:
            nodes (Iterable[str]): The nodes in the schedule.

        Returns:
            Schedule: A new schedule with empty task lists for each node.
        """
        return cls(root={node: [] for node in nodes})

    @computed_field
    @property
    def makespan(self) -> float:
        """Get the makespan of the schedule.

        Returns:
            float: The makespan of the schedule.
        """
        return max((tasks[-1].end for tasks in self.root.values() if tasks), default=0.0)

    def __getitem__(self, node: str) -> List[ScheduledTask]:
        """Get the tasks scheduled on a node.

        Args:
            node (str): The node to get the tasks for.

        Returns:
            List[Task]: A list of tasks scheduled on the node.
        """
        if node not in self.root:
            raise ValueError(f"Node {node} not in schedule. Nodes are {set(self.root.keys())}.")
        return self.root[node]
    
    def items(self):
        """Get the items of the schedule.

        Returns:
            Iterable[Tuple[str, List[ScheduledTask]]]: An iterable of (node, tasks) pairs.
        """
        return self.root.items()

    def get_earliest_start_time(self,
                                node: str,
                                min_start_time: float,
                                exec_time: float) -> float:
        """Get the earliest start time for a task on a node given the minimum start time and execution time.
        
        Args:
            node (str): The node to insert the task on.
            min_start_time (float): The minimum start time of the task.
            exec_time (float): The execution time of the task.
            
        Returns:
            float: The earliest start time of the task.
        """
        if node not in self.root:
            raise ValueError(f"Node {node} not in schedule. Nodes are {set(self.root.keys())}.")
        if not self.root[node] or min_start_time + exec_time <= self.root[node][0].start:
            return min_start_time
        for left, right in zip(self.root[node], self.root[node][1:]):
            if min_start_time >= left.end and min_start_time + exec_time <= right.start:
                return min_start_time
            elif min_start_time < left.end and left.end + exec_time <= right.start:
                return left.end
        return max(min_start_time, self.root[node][-1].end)

    def add_task(self, task: ScheduledTask) -> None:
        """Add a task to the schedule.

        Args:
            task (Task): The task to add.
        """
        if task.node not in self.root:
            raise ValueError(f"Node {task.node} not in schedule. Nodes are {set(self.root.keys())}.")
        idx = bisect.bisect_left([t.start for t in self.root[task.node]], task.start)
        self.root[task.node].insert(idx, task)
        # check that next task starts after this task ends
        if idx + 1 < len(self.root[task.node]) and self.root[task.node][idx + 1].start < task.end:
            raise ValueError(f"Task {task} overlaps with next task {self.root[task.node][idx + 1]}.")
        
    def get_schedulers(self,
                       recursive: bool = False,
                       concrete_only: bool = True) -> Generator[Type['Scheduler'], None, None]:
        """Get all schedulers in the hierarchy.

        Args:
            recursive (bool): Whether to include schedulers from sub-classes.
            concrete_only (bool): Whether to include only concrete schedulers.

        Yields:
            Type['Scheduler']: A scheduler class.
        """
        from saga.scheduler import Scheduler
        for subclass in Scheduler.__subclasses__():
            if not concrete_only or not hasattr(subclass, '__abstractmethods__') or not subclass.__abstractmethods__:
                yield subclass 
            if recursive:
                yield from subclass().get_loaded_schedulers()

    def __str__(self) -> str:
        result = "Schedule:\n"
        for node, tasks in self.root.items():
            result += f"  Node {node}:\n"
            for task in tasks:
                result += f"    {task}\n"
        result += f"  Makespan: {self.makespan:0.2f}\n"
        return result

class NetworkNode(BaseModel):
    """A node in the network."""
    model_config = {'frozen': False}

    name: str = Field(..., description="The name of the node.")
    speed: float = Field(..., description="The speed of the node.")

    def __hash__(self) -> int:
        return hash(self.name)

class NetworkEdge(BaseModel):
    """An edge in the network."""
    model_config = {'frozen': False}

    source: str = Field(..., description="The source node of the edge.")
    target: str = Field(..., description="The target node of the edge.")
    speed: float = Field(default=1e-9, description="The speed (bandwidth) of the edge.")

    def __hash__(self) -> int:
        return hash((self.source, self.target))

    def model_post_init(self, __context: Any) -> None:
        """Set default speed based on source and target if not explicitly provided."""
        if self.speed == 0.0 and self.source == self.target:
            object.__setattr__(self, 'speed', math.inf)

class Network(BaseModel):
    """A network of nodes and edges."""
    model_config = {'frozen': True, 'arbitrary_types_allowed': False}

    nodes: FrozenSet[NetworkNode] = Field(..., description="The nodes in the network.")
    edges: FrozenSet[NetworkEdge] = Field(..., description="The edges in the network.")

    @classmethod
    def create(cls,
               nodes: Iterable[NetworkNode | Tuple[str, float]],
               edges: Iterable[NetworkEdge | Tuple[str, str, float] | Tuple[str, str]]) -> 'Network':
        """Create a new network from nodes and edges.

        Args:
            nodes: An iterable of NetworkNode objects or tuples (name, speed).
            edges: An iterable of NetworkEdge objects or tuples (source, target) or (source, target, speed).

        Returns:
            Network: A new network instance.
        """
        # Create Network
        # - ensure there are not duplicate edges
        # - all edges are undirected
        # - should be fully connected, if not add all missing edges (even self edges)
        # - if edge speed is not provided, default to 1e-9
        # - set all self edges (unless explicitly provided) to 1e9 (speed infinity)
        node_set: Set[NetworkNode] = set()
        for n in nodes:
            if isinstance(n, NetworkNode):
                node_set.add(n)
            elif isinstance(n, tuple) and len(n) == 2:
                node_set.add(NetworkNode(name=n[0], speed=n[1]))
            else:
                raise ValueError(f"Invalid node: {n}")
            
        edge_set: Set[NetworkEdge] = set()
        for e in edges:
            if isinstance(e, NetworkEdge):
                edge_set.add(e)
                edge_set.add(NetworkEdge(source=e.target, target=e.source, speed=e.speed))
            elif isinstance(e, tuple) and len(e) == 3:
                edge_set.add(NetworkEdge(source=e[0], target=e[1], speed=e[2]))
                edge_set.add(NetworkEdge(source=e[1], target=e[0], speed=e[2]))
            elif isinstance(e, tuple) and len(e) == 2:
                edge_set.add(NetworkEdge(source=e[0], target=e[1], speed=1e-9))
                edge_set.add(NetworkEdge(source=e[1], target=e[0], speed=1e-9))
            else:
                raise ValueError(f"Invalid edge: {e}")
            
        for n1, n2 in combinations(node_set, 2):
            if not any(e.source == n1.name and e.target == n2.name for e in edge_set):
                edge_set.add(NetworkEdge(source=n1.name, target=n2.name, speed=1e-9))
                edge_set.add(NetworkEdge(source=n2.name, target=n1.name, speed=1e-9))
        for n in node_set:
            if not any(e.source == n.name and e.target == n.name for e in edge_set):
                edge_set.add(NetworkEdge(source=n.name, target=n.name, speed=1e9))

        return cls(nodes=frozenset(node_set), edges=frozenset(edge_set))

    @cached_property
    def graph(self) -> nx.Graph:
        """Convert the network to a NetworkX graph.

        Returns:
            nx.Graph: The NetworkX graph representation of the network.
        """
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node.name, speed=node.speed)
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, speed=edge.speed)
        return G
    
    @classmethod
    def from_networkx(cls, G: nx.Graph, node_speed_attr: str = "speed", edge_speed_attr: str = "speed") -> 'Network':
        """Create a Network from a NetworkX graph.

        Args:
            G (nx.Graph): The NetworkX graph.
            node_speed_attr (str, optional): The attribute name for the speed of nodes. Defaults to "speed".
            edge_speed_attr (str, optional): The attribute name for the speed of edges. Defaults to "speed".

        Returns:
            Network: The Network representation of the graph.
        """
        nodes = [NetworkNode(name=n, speed=G.nodes[n][node_speed_attr]) for n in G.nodes]
        edges = [NetworkEdge(source=u, target=v, speed=G.edges[u, v][edge_speed_attr]) for u, v in G.edges]
        return cls(nodes=frozenset(nodes), edges=frozenset(edges))
    
    def get_node(self, name: str) -> NetworkNode:
        """Get a node by name.

        Args:
            name (str): The name of the node.

        Returns:
            NetworkNode: The node with the given name.

        Raises:
            ValueError: If the node does not exist.
        """
        for node in self.nodes:
            if node.name == name:
                return node
        raise ValueError(f"Node {name} does not exist in the network.")

    def get_edge(self, source: str, target: str) -> NetworkEdge:
        """Get the edge between two nodes.

        Args:
            source (str): The source node.
            target (str): The target node.

        Returns:
            NetworkEdge: The edge between the source and target nodes.

        Raises:
            ValueError: If the edge does not exist.
        """
        edge = self.graph.get_edge_data(source, target)
        if edge is None:
            raise ValueError(f"Edge from {source} to {target} does not exist.")
        return NetworkEdge(source=source, target=target, speed=edge['speed'])

class TaskGraphNode(BaseModel):
    """A task in the task graph."""
    model_config = {'frozen': False}

    name: str = Field(..., description="The name of the task.")
    cost: float = Field(..., description="The cost (computation time) of the task.")

    def __hash__(self) -> int:
        return hash(self.name)
    
class TaskGraphEdge(BaseModel):
    """A dependency edge in the task graph."""
    model_config = {'frozen': False}

    source: str = Field(..., description="The source task of the edge.")
    target: str = Field(..., description="The target task of the edge.")
    size: float = Field(..., description="The data size of the edge.")

    def __hash__(self) -> int:
        return hash((self.source, self.target))
    
class TaskGraph(BaseModel):
    """A task graph of tasks and dependencies."""
    model_config = {'frozen': True, 'arbitrary_types_allowed': False}

    tasks: FrozenSet[TaskGraphNode] = Field(..., description="The tasks in the task graph.")
    dependencies: FrozenSet[TaskGraphEdge] = Field(..., description="The dependencies in the task graph.")

    @classmethod
    def create(cls,
               tasks: Iterable[TaskGraphNode | Tuple[str, float]],
               dependencies: Iterable[TaskGraphEdge | Tuple[str, str, float] | Tuple[str, str]]) -> 'TaskGraph':
        """Create a new task graph from tasks and dependencies.

        Args:
            tasks: An iterable of TaskGraphNode objects or tuples (name, weight).
            dependencies: An iterable of TaskGraphEdge objects or tuples (source, target) or (source, target, weight).

        Returns:
            TaskGraph: A new task graph instance.
        """
        task_set: Set[TaskGraphNode] = set()
        for t in tasks:
            if isinstance(t, TaskGraphNode):
                task_set.add(t)
            elif isinstance(t, tuple) and len(t) == 2:
                task_set.add(TaskGraphNode(name=t[0], cost=t[1]))
            else:
                raise ValueError(f"Invalid task: {t}")
        dependency_set: Set[TaskGraphEdge] = set()
        for d in dependencies:
            if isinstance(d, TaskGraphEdge):
                dependency_set.add(d)
            elif isinstance(d, tuple) and len(d) == 3:
                dependency_set.add(TaskGraphEdge(source=d[0], target=d[1], size=d[2]))
            elif isinstance(d, tuple) and len(d) == 2:
                dependency_set.add(TaskGraphEdge(source=d[0], target=d[1], size=0.0))
            else:
                raise ValueError(f"Invalid dependency: {d}")

        return cls(tasks=frozenset(task_set), dependencies=frozenset(dependency_set))

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
    def from_networkx(cls, G: nx.DiGraph, node_weight_attr: str = "weight", edge_weight_attr: str = "weight") -> 'TaskGraph':
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
        return cls(tasks=frozenset(tasks), dependencies=frozenset(dependencies))
    
    def topological_sort(self) -> List[TaskGraphNode]:
        """Get a topological sort of the task graph.

        Returns:
            List[TaskGraphNode]: A list of tasks in topological order.
        """
        sorted_task_names = list(nx.topological_sort(self.graph))
        name_to_task = {task.name: task for task in self.tasks}
        return [name_to_task[name] for name in sorted_task_names]
    
    def in_degree(self, task_name: str) -> int:
        """Get the in-degree of a task.

        Args:
            task_name (str): The name of the task.
        Returns:
            int: The in-degree of the task.
        """
        return self.graph.in_degree(task_name)
    
    def in_edges(self, task_name: str) -> List[TaskGraphEdge]:
        """Get the incoming edges of a task.

        Args:
            task_name (str): The name of the task.
        Returns:
            List[TaskGraphEdge]: A list of incoming edges to the task.
        """
        edges = []
        for src, tgt in self.graph.in_edges(task_name):
            edge_data = self.graph.get_edge_data(src, tgt)
            edges.append(TaskGraphEdge(source=src, target=tgt, size=edge_data['weight']))
        return edges
    
    def out_degree(self, task_name: str) -> int:
        """Get the out-degree of a task.

        Args:
            task_name (str): The name of the task.
        Returns:
            int: The out-degree of the task.
        """
        out_degree = self.graph.out_degree(task_name)
        print(f"Out degree of task {task_name}: {out_degree}")  # Debug print
        return out_degree
        return self.graph.out_degree(task_name)
    
    def out_edges(self, task_name: str) -> List[TaskGraphEdge]:
        """Get the outgoing edges of a task.

        Args:
            task_name (str): The name of the task.
        Returns:
            List[TaskGraphEdge]: A list of outgoing edges from the task.
        """
        edges = []
        for src, tgt in self.graph.out_edges(task_name):
            edge_data = self.graph.get_edge_data(src, tgt)
            edges.append(TaskGraphEdge(source=src, target=tgt, size=edge_data['weight']))
        return edges
    
    def get_task(self, name: str) -> TaskGraphNode:
        """Get a task by name.

        Args:
            name (str): The name of the task.

        Returns:
            TaskGraphNode: The task with the given name.

        Raises:
            ValueError: If the task does not exist.
        """
        for task in self.tasks:
            if task.name == name:
                return task
        raise ValueError(f"Task {name} does not exist in the task graph.")
    
    def get_dependency(self, source: str, target: str) -> TaskGraphEdge:
        """Get the dependency edge between two tasks.

        Args:
            source (str): The source task.
            target (str): The target task.

        Returns:
            TaskGraphEdge: The dependency edge between the source and target tasks.

        Raises:
            ValueError: If the dependency does not exist.
        """
        edge = self.graph.get_edge_data(source, target)
        if edge is None:
            raise ValueError(f"Dependency from {source} to {target} does not exist.")
        return TaskGraphEdge(source=source, target=target, size=edge['weight'])

class Scheduler(ABC): # pylint: disable=too-few-public-methods
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
