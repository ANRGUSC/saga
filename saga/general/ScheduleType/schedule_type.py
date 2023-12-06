from abc import ABC, abstractmethod
from typing import Dict, Hashable, List
import networkx as nx

from saga.scheduler import Task

class ScheduleType(ABC): # pylint: disable=too-few-public-methods
    """An abstract class for a scheduler."""
    @abstractmethod
    def insert(self, node: Hashable, task_name: Hashable, comp_schedule: Dict[Hashable, List[Task]], task_schedule: Dict[Hashable, Task]) -> None:
    
        raise NotImplementedError
    
    @abstractmethod
    def get_earliest_finish(self, node: Hashable, task_name: Hashable, comp_schedule: Dict[Hashable, List[Task]], task_schedule: Dict[Hashable, Task]) -> float:

        raise NotImplementedError
    
    @abstractmethod
    def get_earliest_start(self, node: Hashable, task_name: Hashable, comp_schedule: Dict[Hashable, List[Task]], task_schedule: Dict[Hashable, Task]) ->float:

        raise NotImplementedError