from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, List, TypeVar

from pydantic import BaseModel, Field


from saga import (
    NetworkNode,
    Scheduler,
    ScheduledTask,
    Schedule,
    Network,
    TaskGraph,
    TaskGraphNode,
)


class IntialPriority(BaseModel, ABC):
    @abstractmethod
    def call(self, network: Network, task_graph: TaskGraph) -> List[str]:
        """Return the initial priority of the tasks.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            List[str]: The initial priority of the tasks.
        """
        pass


class InsertTask(BaseModel, ABC):
    @abstractmethod
    def call(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Schedule,
        task: str | TaskGraphNode,
        min_start_time: float = 0.0,
        nodes: Iterable[str] | Iterable[NetworkNode] | None = None,
        dry_run: bool = False,
    ) -> ScheduledTask:
        """Insert a task into the schedule.

        Args:
            network (Network): The network graph.
            task_graph (TaskGraph): The task graph.
            schedule (Schedule): The current schedule.
            task (str | TaskGraphNode): The task to insert.
            min_start_time (float, optional): The current moment in time. Defaults to 0.0.
            nodes (Iterable[str] | Iterable[NetworkNode] | None, optional): The nodes to consider for insertion. Defaults to None (all nodes).
            dry_run (bool, optional): If True, do not modify the schedule. Defaults to False.

        Returns:
            Task: The inserted task
        """
        pass


TInsert = TypeVar("TInsert", bound=InsertTask)


class ParametricScheduler(Scheduler, BaseModel, Generic[TInsert]):
    initial_priority: IntialPriority = Field(
        ..., description="The initial priority strategy."
    )
    insert_task: TInsert = Field(..., description="The task insertion strategy.")

    def __init__(
        self, initial_priority: IntialPriority, insert_task: TInsert, **kwargs: Any
    ) -> None:
        super().__init__(
            initial_priority=initial_priority, insert_task=insert_task, **kwargs
        )

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Schedule | None = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Schedule the tasks on the network.

        Args:
            network (Network): The network graph.
            task_graph (TaskGraph): The task graph.
            schedule (Schedule): The current schedule.
            min_start_time (float): The current moment in time.

        Returns:
            Schedule: The resulting schedule.
        """
        if schedule is None:
            schedule = Schedule(task_graph, network)
        queue = self.initial_priority.call(network, task_graph)
        while queue:
            task_name = queue.pop(0)
            if schedule.is_scheduled(task_name):
                continue
            else:
                self.insert_task.call(
                    network, task_graph, schedule, task_name, min_start_time
                )
        return schedule

    @property
    def name(self) -> str:
        return f"{self.insert_task.__class__.__name__}_{self.initial_priority.__class__.__name__}"
