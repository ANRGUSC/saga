from abc import ABC, abstractmethod
import math
from typing import Any, Dict, FrozenSet, Generator, Hashable, Iterable, List, Optional, Set, Tuple, Type
import networkx as nx
from pydantic import BaseModel, RootModel, computed_field, Field
import bisect
from itertools import combinations

class Task(BaseModel):
    """A scheduled task."""
    node: Hashable = Field(..., description="The node the task is scheduled on.")
    name: Hashable = Field(..., description="The name of the task.")
    start: float = Field(..., description="The start time of the task.")
    end: float = Field(..., description="The end time of the task.")

    def __str__(self) -> str:
        return f"Task(node={self.node}, name={self.name}, start={self.start:0.2f}, end={self.end:0.2f})"

class Schedule(RootModel[Dict[Hashable, List[Task]]]):
    """A schedule of tasks on nodes."""

    root: Dict[Hashable, List[Task]]

    @classmethod
    def create(cls, nodes: Iterable[Hashable]) -> 'Schedule':
        """Create a new schedule.

        Args:
            nodes (Iterable[Hashable]): The nodes in the schedule.

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

    def __getitem__(self, node: Hashable) -> List[Task]:
        """Get the tasks scheduled on a node.

        Args:
            node (Hashable): The node to get the tasks for.

        Returns:
            List[Task]: A list of tasks scheduled on the node.
        """
        if node not in self.root:
            raise ValueError(f"Node {node} not in schedule. Nodes are {set(self.root.keys())}.")
        return self.root[node]

    def get_earliest_start_time(self,
                                node: Hashable,
                                min_start_time: float,
                                exec_time: float) -> float:
        """Get the earliest start time for a task on a node given the minimum start time and execution time.
        
        Args:
            node (Hashable): The node to insert the task on.
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

    def add_task(self, task: Task) -> None:
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

class NetworkNode(BaseModel):
    """A node in the network."""
    model_config = {'frozen': False}

    name: Hashable = Field(..., description="The name of the node.")
    speed: float = Field(..., description="The speed of the node.")

    def __hash__(self) -> int:
        return hash(self.name)

class NetworkEdge(BaseModel):
    """An edge in the network."""
    model_config = {'frozen': False}

    source: Hashable = Field(..., description="The source node of the edge.")
    target: Hashable = Field(..., description="The target node of the edge.")
    speed: float = Field(default=0.0, description="The speed (bandwidth) of the edge.")

    def __hash__(self) -> int:
        return hash((self.source, self.target))

    def model_post_init(self, __context: Any) -> None:
        """Set default speed based on source and target if not explicitly provided."""
        if self.speed == 0.0 and self.source == self.target:
            object.__setattr__(self, 'speed', math.inf)

class Network(BaseModel):
    """A network of nodes and edges."""
    model_config = {'frozen': False, 'arbitrary_types_allowed': True}

    nodes: FrozenSet[NetworkNode] = Field(..., description="The nodes in the network.")
    edges: FrozenSet[NetworkEdge] = Field(..., description="The edges in the network.")

    @classmethod
    def create(cls,
               nodes: Iterable[NetworkNode | Tuple[Hashable, float]],
               edges: Iterable[NetworkEdge | Tuple[Hashable, Hashable, float] | Tuple[Hashable, Hashable]]) -> 'Network':
        """Create a new network from nodes and edges.

        Args:
            nodes: An iterable of NetworkNode objects or tuples (name, speed).
            edges: An iterable of NetworkEdge objects or tuples (source, target) or (source, target, speed).

        Returns:
            Network: A new network instance.
        """
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
            elif isinstance(e, tuple) and len(e) == 3:
                edge_set.add(NetworkEdge(source=e[0], target=e[1], speed=e[2]))
            elif isinstance(e, tuple) and len(e) == 2:
                edge_set.add(NetworkEdge(source=e[0], target=e[1]))
            else:
                raise ValueError(f"Invalid edge: {e}")

        # for any edge that does not exist, add the default edge
        existing_edges = {(edge.source, edge.target) for edge in edge_set}
        for u, v in combinations(node_set, 2):
            if (u.name, v.name) not in existing_edges:
                edge_set.add(NetworkEdge(source=u.name, target=v.name))
            if (v.name, u.name) not in existing_edges:
                edge_set.add(NetworkEdge(source=v.name, target=u.name))
        for n in node_set:
            if (n.name, n.name) not in existing_edges:
                edge_set.add(NetworkEdge(source=n.name, target=n.name))

        return cls(nodes=frozenset(node_set), edges=frozenset(edge_set))


    def to_networkx(self) -> nx.Graph:
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


class Scheduler(ABC): # pylint: disable=too-few-public-methods
    """An abstract class for a scheduler."""

    @abstractmethod
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Schedule:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: A dictionary mapping nodes to a list of tasks executed on the node.
        """
        raise NotImplementedError
    
    def get_loaded_schedulers(self) -> List[Type['Scheduler']]:
        """Get the list of loaded schedulers.

        Returns:
            List[Type['Scheduler']]: A list of loaded schedulers.
        """
        return Scheduler.__subclasses__()
