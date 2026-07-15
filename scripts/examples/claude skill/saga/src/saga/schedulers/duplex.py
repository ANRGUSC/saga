from typing import Optional

from saga import Network, Schedule, Scheduler, TaskGraph
from saga.schedulers.maxmin import MaxMinScheduler
from saga.schedulers.minmin import MinMinScheduler


class DuplexScheduler(Scheduler):
    """Duplex scheduler"""

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Returns the best schedule (minimizing makespan) for a problem instance using duplex

        Args:
            network: Network
            task_graph: Task graph
            schedule: Optional initial schedule to build upon. Defaults to None.
            min_start_time: Minimum start time for tasks. Defaults to 0.0.

        Returns:
            A Schedule object containing the computed schedule.
        """
        minmin_schedule = MinMinScheduler().schedule(
            network, task_graph, schedule, min_start_time
        )
        maxmin_schedule = MaxMinScheduler().schedule(
            network, task_graph, schedule, min_start_time
        )

        if minmin_schedule.makespan <= maxmin_schedule.makespan:
            return minmin_schedule
        return maxmin_schedule
