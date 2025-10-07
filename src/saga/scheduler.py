from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Hashable, List, Optional, Tuple, Type
import networkx as nx
from pydantic import BaseModel, RootModel
from pydantic.main import TupleGenerator

class Task(BaseModel):
    """A task."""
    node: Hashable
    name: Hashable
    start: float
    end: float

class Schedule(RootModel[Dict[Hashable, List[Task]]]): # pylint: disable=too-few-public-methods
    """A schedule of nodes with tasks assigned to them."""
    @property
    def makespan(self) -> float:
        """Get the makespan of the schedule.

        Returns:
            float: The makespan of the schedule.
        """
        return max(task.end for tasks in self.root.values() for task in tasks if task.end is not None)
    
    def __iter__(self) -> Generator[Tuple[Hashable, List[Task]], None, None]: # type: ignore
        """Iterate over the schedule.

        Yields:
            Generator[Tuple[Hashable, List[Task]], None, None]: A generator of tuples of node and list of tasks.
        """
        yield from self.root.items()

    def __getitem__(self, key: Hashable) -> List[Task]: # type: ignore
        """Get the tasks for a node.

        Args:
            key (Hashable): The node.
        Returns:
            List[Task]: The list of tasks for the node.
        Raises:
            KeyError: If the node is not in the schedule.
        """
        return self.root[key]
    
    def add_task(self, task: Task) -> None:
        """Add a task to the schedule.

        Args:
            task (Task): The task to add.
        """
        if task.node not in self.root:
            self.root[task.node] = []
        self.root[task.node].append(task)

    # TODO: enforce property that task.node matches the key in the dict

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
