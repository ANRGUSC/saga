from abc import ABC, abstractmethod
from typing import Dict, Hashable, List
import networkx as nx
from saga.scheduler import Task


class ScheduleType(ABC):  # pylint: disable=too-few-public-methods
    """An abstract class for a scheduler."""

    @abstractmethod
    def insert(
        self,
        node: Hashable,
        task_name: Hashable,
        comp_schedule: Dict[Hashable, List[Task]],
        task_schedule: Dict[Hashable, Task],
    ) -> None:
        """
        Insert a task into the schedule IN-PLACE.
        Args:
            node (Hashable): The node to schedule the task on.
            task_name (Hashable): The name of the task to schedule.
            comp_schedule (Dict[Hashable, List[Task]]): The current computation schedule.
            task_schedule (Dict[Hashable, Task]): The current task schedule.
        """
        raise NotImplementedError

    @abstractmethod
    def get_earliest_finish(
        self,
        node: Hashable,
        task_name: Hashable,
        comp_schedule: Dict[Hashable, List[Task]],
        task_schedule: Dict[Hashable, Task],
    ) -> float:
        """
        Get the earliest finish time of a task on a node.
        Args:
            node (Hashable): The node to schedule the task on.
            task_name (Hashable): The name of the task to schedule.
            comp_schedule (Dict[Hashable, List[Task]]): The current computation schedule.
            task_schedule (Dict[Hashable, Task]): The current task schedule.
        Returns:
            float: The earliest finish time of the task on the node.
        """
        raise NotImplementedError

    @abstractmethod
    def get_earliest_start(
        self,
        node: Hashable,
        task_name: Hashable,
        comp_schedule: Dict[Hashable, List[Task]],
        task_schedule: Dict[Hashable, Task],
    ) -> float:
        """
        Get the earliest start time of a task on a node.
        Args:
            node (Hashable): The node to schedule the task on.
            task_name (Hashable): The name of the task to schedule.
            comp_schedule (Dict[Hashable, List[Task]]): The current computation schedule.
            task_schedule (Dict[Hashable, Task]): The current task schedule.
        Returns:
            float: The earliest start time of the task on the node.
        """
        raise NotImplementedError

    @classmethod
    def get_selection_metrics(cls) -> Dict[str, Callable]:
        """
        Get the selection metrics for the scheduler.
        Returns:
            Dict[str, Callable]: The selection metrics for the scheduler.
        """
        return {
            "earliest_finish": cls.get_earliest_finish,
            "earliest_start": cls.get_earliest_start,
        }

    def schedule(network: nx.Graph,
                 ranking: List[Hashable],
                 selection_metric: str) -> Dict[Hashable, List[Task]]:
        selection_metrics = ScheduleType.get_selection_metrics()
        selection_metric = selection_metrics[selection_metric]
        
        comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}
        for task_name in ranking:
            min_metric_value = np.inf
            best_node = None
            for node in network.nodes:  # Find the best node to run the task
                metric_value = selection_metric(node, task_name, comp_schedule, task_schedule)
                if metric_value < min_metric_value:
                    min_metric_value = metric_value
                    best_node = node
            
            scheduler.insert(best_node, task_name, comp_schedule, task_schedule)
        
        return comp_schedule