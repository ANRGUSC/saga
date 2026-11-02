import pathlib
from typing import List, Optional
import numpy as np

from saga import Schedule, Scheduler, ScheduledTask, TaskGraph, Network
from saga.schedulers.topographic_random_selection import random_topographic_sort
thisdir = pathlib.Path(__file__).resolve().parent

class RandomHeftScheduler(Scheduler):
    """Schedules tasks using a modified HEFT algorithm, where the upranking step is replaced
    by a random topographic sort.

    Source: https://dx.doi.org/10.1109/71.993206
    """

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (Optional[Schedule], optional): An existing partial schedule to
                extend. Defaults to None.
            min_start_time (float, optional): The minimum start time. Defaults to 0.0.

        Returns:
            Schedule: The resulting schedule.

        Raises:
            ValueError: If the instance is invalid.
        """
        initial_schedule_order = random_topographic_sort(task_graph)
        schedule_order = list()
        for task in initial_schedule_order:
            task_name = task.name
            schedule_order.append(task_name)
        schedule = schedule if schedule is not None else Schedule(task_graph, network)

        for task_name in schedule_order:
            if schedule.is_scheduled(task_name):
                continue
            min_finish_time = np.inf
            best_node = next(iter(network.nodes))  # arbitrary initialization
            for node in network.nodes:
                start_time = schedule.get_earliest_start_time(
                    task=task_name,
                    node=node,
                    append_only=False,
                    current_moment=min_start_time,
                )
                runtime = (
                    task_graph.get_task(task_name).cost / network.get_node(node).speed
                )
                finish_time = start_time + runtime
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_node = node
            new_task = ScheduledTask(
                node=best_node.name,
                name=task_name,
                start=min_finish_time
                - (
                    task_graph.get_task(task_name).cost
                    / network.get_node(best_node).speed
                ),
                end=min_finish_time,
            )
            schedule.add_task(new_task)

        return schedule
