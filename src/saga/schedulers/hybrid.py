from typing import Iterable
from saga import Schedule, Scheduler, TaskGraph, Network


class HybridScheduler(Scheduler):
    """A hybrid scheduler."""

    def __init__(self, schedulers: Iterable[Scheduler]) -> None:
        """Initializes the hybrid scheduler.

        Args:
            schedulers (Iterable[Scheduler]): An iterable of schedulers.
        """
        self.schedulers = schedulers

    def schedule(self, network: Network, task_graph: TaskGraph) -> Schedule:
        """Returns the best schedule of the given schedule functions.

        Args:
            network (Network): The network graph.
            task_graph (TaskGraph): The task graph.

        Returns:
            Dict[str, List[Task]]: The best schedule.
        #"""
        return min(
            (scheduler.schedule(network, task_graph) for scheduler in self.schedulers),
            key=lambda s: s.makespan,
        )
