from abc import ABC, abstractmethod
from typing import Dict, Hashable, List

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
