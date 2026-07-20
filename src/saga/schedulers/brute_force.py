import itertools
from copy import deepcopy
from typing import Optional

from saga import Scheduler, ScheduledTask, TaskGraph, Network, Schedule


class BruteForceScheduler(Scheduler):
    """Brute force scheduler"""

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Returns the best schedule (minimizing makespan) for a problem
           instance using brute force

        Args:
            network: Network
            task_graph: Task graph
            schedule: An existing partial schedule to extend. Defaults to None.
            min_start_time: The earliest time newly placed tasks may start.

        Returns:
            Schedule: The resulting schedule.
        """
        # get all topological sorts of the task graph
        topological_sorts = list(task_graph.all_topological_sorts())
        # get all valid mappings of the task graph nodes to the network nodes
        mappings = [
            dict(zip(task_graph.tasks, mapping))
            for mapping in itertools.product(
                network.nodes, repeat=len(task_graph.tasks)
            )
        ]

        best_schedule = None
        best_makespan = float("inf")
        for mapping in mappings:
            for top_sort in topological_sorts:
                candidate_schedule = (
                    deepcopy(schedule)
                    if schedule is not None
                    else Schedule(task_graph, network)
                )
                for task in top_sort:
                    if candidate_schedule.is_scheduled(task.name):
                        continue
                    node = mapping[task]
                    ready_time = candidate_schedule.get_earliest_start_time(
                        task.name,
                        node.name,
                        append_only=True,
                        current_moment=min_start_time,
                    )
                    new_task = ScheduledTask(
                        node=node.name,
                        name=task.name,
                        start=ready_time,
                        end=ready_time + task.cost / node.speed,
                    )
                    candidate_schedule.add_task(new_task)

                if candidate_schedule.makespan < best_makespan:
                    best_makespan = candidate_schedule.makespan
                    best_schedule = candidate_schedule

        if best_schedule is None:
            raise RuntimeError("Brute force scheduler failed to find a schedule")
        return best_schedule
