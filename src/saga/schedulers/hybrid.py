from copy import deepcopy
from typing import List, Optional
from pydantic import Field
from saga import Schedule, Scheduler, TaskGraph, Network


class HybridScheduler(Scheduler):
    """A hybrid scheduler."""

    schedulers: List[Scheduler] = Field(...)

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Returns the best schedule of the given schedule functions.

        Args:
            network (Network): The network graph.
            task_graph (TaskGraph): The task graph.

        Returns:
            Dict[str, List[Task]]: The best schedule.
        #"""
        # Each child gets its own copy of the partial schedule: Schedule is mutable,
        # so sharing one instance would let each scheduler see the previous one's
        # placements.
        return min(
            (
                scheduler.schedule(
                    network,
                    task_graph,
                    deepcopy(schedule) if schedule is not None else None,
                    min_start_time,
                )
                for scheduler in self.schedulers
            ),
            key=lambda s: s.makespan,
        )
