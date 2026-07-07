"""ReschedulePolicy: re-plan remaining tasks around committed ones every step."""
from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from saga import Schedule
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
            environment.schedule_actual = new_schedule
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
