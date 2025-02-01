from abc import ABC, abstractmethod
from copy import deepcopy
import logging
from typing import Any, Dict, Hashable, List, Optional, Tuple
import networkx as nx

from saga.scheduler import Scheduler, Task

class IntialPriority(ABC):
    @abstractmethod
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> List[Hashable]:
        """Return the initial priority of the tasks.
        
        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            List[Hashable]: The initial priority of the tasks.
        """
        pass

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Return a dictionary representation of the initial priority.
        
        Returns:
            Dict[str, Any]: A dictionary representation of the initial priority.
        """
        pass

    @classmethod
    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> "IntialPriority":
        """Return a new instance of the initial priority from the serialized data.
        
        Args:
            data (Dict[str, Any]): The serialized data.

        Returns:
            IntialPriority: A new instance of the initial priority.
        """
        pass

ScheduleType = Dict[Hashable, List[Task]]
class InsertTask(ABC):
    @abstractmethod
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: ScheduleType,
                 task: Hashable,
                 node: Optional[Hashable] = None) -> Task:
        """Insert a task into the schedule.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (ScheduleType): The schedule.
            task (Hashable): The task.  
            node (Optional[Hashable]): The node to insert the task onto. If None, the node is chosen by the algorithm.

        Returns:
            Task: The inserted task
        """
        pass

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Return a dictionary representation of the initial priority.
        
        Returns:
            Dict[str, Any]: A dictionary representation of the initial priority.
        """
        pass

    @classmethod
    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> "InsertTask":
        """Return a new instance of the initial priority from the serialized data.
        
        Args:
            data (Dict[str, Any]): The serialized data.

        Returns:
            InsertTask: A new instance of the initial priority.
        """
        pass

class ParametricScheduler(Scheduler):
    def __init__(self,
                 initial_priority: IntialPriority,
                 insert_task: InsertTask) -> None:
            super().__init__()
            self.initial_priority = initial_priority
            self.insert_task = insert_task
    
    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: Optional[ScheduleType] = None,
                 min_start_time: float = 0.0) -> ScheduleType:
        """Schedule the tasks on the network.
        
        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (Optional[ScheduleType]): The current schedule.
            min_start_time: float

        Returns:
            Dict[Hashable, List[Task]]: A dictionary mapping nodes to a list of tasks executed on the node.
        """
        if schedule is not None:
            schedule = {
                node: [
                    Task(task.node, task.name, task.start - min_start_time, task.end - min_start_time)
                    for task in tasks
                ]
                for node, tasks in schedule.items()
            }
        queue = self.initial_priority(network, task_graph)
        schedule = {node: [] for node in network.nodes} if schedule is None else deepcopy(schedule)
        while queue:
            self.insert_task(network, task_graph, schedule, queue.pop(0))

        schedule = {
            node: [
                Task(task.node, task.name, task.start + min_start_time, task.end + min_start_time)
                for task in tasks
            ]
            for node, tasks in schedule.items()
        }
        return schedule

    def serialize(self) -> Dict[str, Any]:
        """Return a dictionary representation of the initial priority.
        
        Returns:
            Dict[str, Any]: A dictionary representation of the initial priority.
        """
        return {
            "name": "ParametricScheduler",
            "initial_priority": self.initial_priority.serialize(),
            "insert_task": self.insert_task.serialize(),
            "k_depth": 0
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "ParametricScheduler":
        """Return a new instance of the initial priority from the serialized data.
        
        Args:
            data (Dict[str, Any]): The serialized data.

        Returns:
            ParametricScheduler: A new instance of the initial priority.
        """
        return cls(
            initial_priority=IntialPriority.deserialize(data["initial_priority"]),
            insert_task=InsertTask.deserialize(data["insert_task"])
        )
