from abc import ABC, abstractmethod
from copy import deepcopy
import logging
from typing import Any, Dict, Hashable, List, Optional, Tuple
import networkx as nx
from saga.utils.online_tools import ScheduleInjector

from saga import Scheduler, ScheduledTask

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

ScheduleType = Dict[Hashable, List[ScheduledTask]]
class InsertTask(ABC): 
    @abstractmethod
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: ScheduleType,
                 #task_map: Dict[Hashable, Task],
                 task: Hashable,
                 current_moment: float,
    
                 node: Optional[Hashable] = None) -> ScheduledTask:
        
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

class ParametricScheduler(ScheduleInjector, Scheduler):
    def __init__(self,
                 initial_priority: IntialPriority,
                 insert_task: InsertTask) -> None:
            super().__init__()
            self.initial_priority = initial_priority
            self.insert_task = insert_task
    
    def _do_schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 comp_schedule: ScheduleType,
                 task_map: Dict[Hashable, ScheduledTask],
                 current_moment:float,
                 **_algo_kwargs) -> ScheduleType:
        """Schedule the tasks on the network.
        
        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (Optional[ScheduleType]): The current schedule.

        Returns:
            Dict[Hashable, List[Task]]: A dictionary mapping nodes to a list of tasks executed on the node.
        """
        queue = self.initial_priority(network, task_graph)
        #schedule = {node: [] for node in network.nodes} if schedule is None else deepcopy(schedule)
        while queue: #!!
            task_name = queue.pop(0)
            if task_name in task_map:
                continue
            else:
                self.insert_task(network, task_graph, comp_schedule, task_name, current_moment)
        return comp_schedule

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
