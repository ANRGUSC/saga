from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Hashable, List, Optional, Tuple, Type
import networkx as nx
from pydantic import BaseModel
from pydantic.main import TupleGenerator

class Task(BaseModel):
    """A task."""
    node: str
    name: str
    start: float
    end: float

class Schedule(BaseModel):
    """A schedule."""
    schedule: Dict[Hashable, List[Task]] = {}

    @property
    def makespan(self) -> float:
        """Get the makespan of the schedule.

        Returns:
            float: The makespan of the schedule.
        """
        return max(task.end for tasks in self.schedule.values() for task in tasks if task.end is not None)

    def add_task(self, node: Hashable, task: Task) -> None:
        """Add a task to the schedule.

        Args:
            node (Hashable): The node to add the task to.
            task (Task): The task to add.
        """
        if node not in self.schedule:
            self.schedule[node] = []
        self.schedule[node].append(task)

    def items(self) -> Generator[Tuple[Hashable, List[Task]], None, None]:
        """
        Get the items of the schedule.

        Yields:
            Generator[Tuple[Hashable, List[Task]], None, None]: The items of the schedule.
        """
        yield from self.schedule.items()

    def __getitem__(self, node: Hashable) -> List[Task]:
        """
        Get the tasks for a node.

        Args:
            node (Hashable): The node to get the tasks for.

        Returns:
            List[Task]: The tasks for the node.

        Raises:
            KeyError: If the node is not in the schedule.
        """
        return self.schedule[node]
    
    @property
    def task_map(self) -> Dict[str, Task]:
        """
        Get a mapping of task names to tasks.

        Returns:
            Dict[str, Task]: A mapping of task names to tasks.
        """
        task_map: Dict[str, Task] = {}
        for tasks in self.schedule.values():
            for task in tasks:
                task_map[task.name] = task
        return task_map

    def sort_tasks(self) -> None:
        """Sort the tasks in the schedule by start time."""
        for tasks in self.schedule.values():
            tasks.sort(key=lambda task: task.start if task.start is not None else float('inf'))

    def is_sorted(self) -> bool:
        """Check if the tasks in the schedule are sorted by start time.

        Returns:
            bool: True if the tasks are sorted, False otherwise.
        """
        for tasks in self.schedule.values():
            if any(tasks[i].start > tasks[i+1].start for i in range(len(tasks)-1)):
                return False
        return True
    
    def is_valid(self) -> bool:
        """Check if the schedule is valid in terms of task overlaps.

        Returns:
            bool: True if the schedule is valid, False otherwise.
        """
        for tasks in self.schedule.values():
            for i in range(len(tasks)-1):
                if tasks[i].end > tasks[i+1].start:
                    return False
        return True

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
