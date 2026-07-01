"""Stochastic online environment: tracks an "actual" schedule under realized variance."""
from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING, cast

import numpy as np

from saga import Network, Scheduler, TaskGraph
from saga.schedulers.online.environment import Environment, StepFunction, next_completion
from saga.schedulers.stochastic.estimate_stochastic_scheduler import EstimateStochasticScheduler
from saga.stochastic import StochasticNetwork, StochasticSchedule, StochasticTaskGraph
from saga.utils.random_variable import RandomVariable

if TYPE_CHECKING:
    from saga.schedulers.online.policy import OnlinePolicy


class StochasticEnvironment(Environment):
    """Environment wrapper that adds stochasticity to task execution and network transmission times.

    The base Environment tracks the deterministic "nominal" schedule based on the input
    Network and TaskGraph. The StochasticEnvironment maintains a separate "actual" schedule
    that reflects the realized start/end times of tasks and transmissions under stochasticity.
    The step function and policy can read from either schedule as needed.
    *** WIP ***
    """

    def __init__(
        self,
        network: StochasticNetwork,
        task_graph: StochasticTaskGraph,
        scheduler: Scheduler,
        estimate: Callable[[RandomVariable], float],
        policy: Optional["OnlinePolicy"] = None,
        step: StepFunction = next_completion,
        on_step: Optional[Callable[["Environment"], None]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            network=cast(Network, network),
            task_graph=cast(TaskGraph, task_graph),
            scheduler=scheduler,
            policy=policy,
            step=step,
            on_step=on_step,
        )
        self._stochastic_task_graph = task_graph
        self._stochastic_network = network
        self.seed = seed
        self.actual_task_graph = task_graph.sample()
        self.actual_network = network.sample()
        self._estimate = estimate
        self.stochastic_scheduler = EstimateStochasticScheduler(
            scheduler=scheduler,
            estimate=estimate,
        )

        self.initial_estimate_schedule, self.network, self.task_graph = self.stochastic_scheduler.schedule(
            network=network, task_graph=task_graph
        )
        self.estimate_schedule: StochasticSchedule = self.initial_estimate_schedule
        self.schedule_actual = self.estimate_schedule.determinize(self.actual_network, self.actual_task_graph)

    def reset(self) -> None:
        super().reset()
        if self.seed is not None:
            np.random.seed(self.seed)
        self.actual_task_graph = self._stochastic_task_graph.sample()
        self.actual_network = self._stochastic_network.sample()
        self.initial_estimate_schedule, self.network, self.task_graph = self.stochastic_scheduler.schedule(
            network=self._stochastic_network, task_graph=self._stochastic_task_graph
        )
        self.estimate_schedule = self.initial_estimate_schedule
        self.schedule_actual = self.estimate_schedule.determinize(self.actual_network, self.actual_task_graph)
        self.schedule = self.schedule_actual

    def _update_task_state(self) -> None:
        """Recompute all four task-state sets from the current schedule and task graph."""
        self.schedule_actual = self.estimate_schedule.determinize(self.actual_network, self.actual_task_graph)
        self.schedule = self.schedule_actual
        self.finished_tasks = super().get_finished_tasks(self.schedule_actual)
        self.running_tasks = super().get_running_tasks(self.schedule_actual)

        finished_names = {t.name for t in self.finished_tasks}
        committed_names = finished_names | {t.name for t in self.running_tasks}

        self.ready_tasks = set()
        for tasks in self.schedule_actual.mapping.values():
            for task in tasks:
                if task.name in committed_names:
                    continue
                predecessors = {dep.source for dep in self.task_graph.in_edges(task.name)}
                if predecessors.issubset(finished_names):
                    self.ready_tasks.add(task)
                    self.unready_tasks.discard(task)
