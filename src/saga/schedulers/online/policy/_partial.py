"""Shared helper: build a partial schedule from an environment's committed tasks."""
from __future__ import annotations

from copy import deepcopy
from typing import Set, TYPE_CHECKING

from saga import Schedule, ScheduledTask
from saga.schedulers.online.environment import StochasticEnvironment

if TYPE_CHECKING:
    from saga.schedulers.online.environment import Environment


def build_partial_schedule(environment: "Environment") -> Schedule:
    """Return a Schedule containing only the committed (finished + running) tasks."""
    partial = Schedule(
        environment.task_graph,
        environment.network,
        node_constraints=environment.node_constraints,
    )
    if isinstance(environment, StochasticEnvironment):
        environment.schedule = environment.estimate_schedule.determinize(
            environment.actual_network, environment.actual_task_graph
        )
        for task in environment.finished_tasks:
            partial.add_task(task)

        est_running_tasks: Set[ScheduledTask] = deepcopy(environment.running_tasks)
        for task in est_running_tasks:
            est_task_size = environment.task_graph.get_task(task.name).cost
            est_network_speed = environment.network.get_node(task.node).speed
            task.end = task.start + (est_task_size / est_network_speed)
            partial.add_task(task)
        return partial
    else:
        for task in environment.finished_tasks:
            partial.add_task(task)
        for task in environment.running_tasks:
            partial.add_task(deepcopy(task))
        return partial
