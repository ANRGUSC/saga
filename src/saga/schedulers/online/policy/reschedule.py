"""ReschedulePolicy: re-plan remaining tasks around committed ones every step."""
from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING
import random

from saga import Schedule, ScheduledTask
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.online.policy import OnlinePolicy
from saga.schedulers.online.policy._partial import build_partial_schedule
from saga.schedulers.online.environment import StochasticEnvironment

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from saga.schedulers.online.environment import Environment


class ReschedulePolicy(OnlinePolicy):
    """Reschedules all remaining tasks around committed (finished + running) tasks every step."""

    def update(self, environment: "Environment") -> Optional[Schedule]:
        if not isinstance(environment.scheduler, ParametricScheduler):
            logger.warning(
                "ReschedulePolicy: env.scheduler is not a ParametricScheduler "
                "(%s). Rescheduling requires schedule and min_start_time support; "
                "this may raise at runtime.",
                type(environment.scheduler).__name__,
            )
        partial = build_partial_schedule(environment)
        if isinstance(environment, StochasticEnvironment):
            new_estimate = environment.stochastic_scheduler.schedule(
                environment._stochastic_network,
                environment._stochastic_task_graph,
                schedule=partial,
                min_start_time=environment.current_time,
                node_constraints=environment.node_constraints,
            )[0]
            environment.estimate_schedule = new_estimate
            new_schedule = new_estimate.determinize(
                environment.actual_network, environment.actual_task_graph
            )
            environment.schedule = new_schedule
        else:
            if environment.scheduler is None:
                raise ValueError("ReschedulePolicy requires environment.scheduler to be set.")
            new_schedule = environment.scheduler.schedule(
                environment.network,
                environment.task_graph,
                schedule=partial,
                min_start_time=environment.current_time,
            )
        return new_schedule


class ConditionalReschedulePolicy(OnlinePolicy):
    """Reschedules all remaining tasks around committed (finished + running) tasks at specific points."""
    def evaluate_reschedule(
        self, scheduled_task: ScheduledTask, environment: "StochasticEnvironment"
    ) -> bool:
        """Flag a reschedule when a task's actual exec time is an outlier vs. its estimate.

        Assumes task cost and node speed are independent, so comp_time = cost * (1/speed):
            E[comp_time]   = E[cost] * E[1/speed]
            Var[comp_time] = E[cost^2] * E[1/speed^2] - E[comp_time]^2
        """
        est = environment._estimate
        stochastic_task = environment._stochastic_task_graph.get_task(scheduled_task.name)
        stochastic_processor = environment._stochastic_network.get_node(scheduled_task.node)

        inv_speed = 1 / stochastic_processor.speed
        expected_exec = est(stochastic_task.cost) * est(inv_speed)
        var = (
            est(stochastic_task.cost * stochastic_task.cost) * est(inv_speed * inv_speed)
            - expected_exec**2
        )
        # var can dip slightly negative from Monte Carlo estimation noise near zero;
        # clamp so sqrt never returns a complex number.
        sd = max(var, 0.0) ** 0.5

        exec_time = scheduled_task.end - scheduled_task.start
        return abs(exec_time - expected_exec) > 1.5 * sd


    def update(self, environment: "Environment") -> Optional[Schedule]:
        if not isinstance(environment.scheduler, ParametricScheduler):
            logger.warning(
                "ReschedulePolicy: env.scheduler is not a ParametricScheduler "
                "(%s). Rescheduling requires schedule and min_start_time support; "
                "this may raise at runtime.",
                type(environment.scheduler).__name__,
            )
        last_finished = max(environment.finished_tasks, key=lambda t: t.end)
        if not self.evaluate_reschedule(scheduled_task=last_finished, environment=environment):
            return environment.schedule

        partial = build_partial_schedule(environment)
        if isinstance(environment, StochasticEnvironment):
            new_estimate = environment.stochastic_scheduler.schedule(
                environment._stochastic_network,
                environment._stochastic_task_graph,
                schedule=partial,
                min_start_time=environment.current_time,
                node_constraints=environment.node_constraints,
            )[0]
            environment.estimate_schedule = new_estimate
            new_schedule = new_estimate.determinize(
                environment.actual_network, environment.actual_task_graph
            )
            environment.schedule = new_schedule
        else:
            if environment.scheduler is None:
                raise ValueError("ReschedulePolicy requires environment.scheduler to be set.")
            new_schedule = environment.scheduler.schedule(
                environment.network,
                environment.task_graph,
                schedule=partial,
                min_start_time=environment.current_time,
            )
        return new_schedule


class RandomReschedulePolicy10(OnlinePolicy):

    def evaluate_reschedule(self):
        if random.randint(1,10) == 1:
            return True
        return False


    def update(self, environment: "Environment") -> Optional[Schedule]:
        if not isinstance(environment.scheduler, ParametricScheduler):
            logger.warning(
                "ReschedulePolicy: env.scheduler is not a ParametricScheduler "
                "(%s). Rescheduling requires schedule and min_start_time support; "
                "this may raise at runtime.",
                type(environment.scheduler).__name__,
            )
        last_finished = max(environment.finished_tasks, key=lambda t: t.end)
        if not self.evaluate_reschedule(scheduled_task=last_finished, environment=environment):
            return environment.schedule

        partial = build_partial_schedule(environment)
        if isinstance(environment, StochasticEnvironment):
            new_estimate = environment.stochastic_scheduler.schedule(
                environment._stochastic_network,
                environment._stochastic_task_graph,
                schedule=partial,
                min_start_time=environment.current_time,
                node_constraints=environment.node_constraints,
            )[0]
            environment.estimate_schedule = new_estimate
            new_schedule = new_estimate.determinize(
                environment.actual_network, environment.actual_task_graph
            )
            environment.schedule = new_schedule
        else:
            if environment.scheduler is None:
                raise ValueError("ReschedulePolicy requires environment.scheduler to be set.")
            new_schedule = environment.scheduler.schedule(
                environment.network,
                environment.task_graph,
                schedule=partial,
                min_start_time=environment.current_time,
            )
        return new_schedule
    