"""The OnlinePolicy interface and its concrete implementations.

A policy is the pluggable decision logic of an online algorithm: "if X is
observed, do Y". It is consulted once per simulation step and either returns a
revised :class:`~saga.Schedule` or ``None`` to leave the schedule unchanged.
Concrete policies live in sibling modules and are re-exported here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from saga import Schedule

if TYPE_CHECKING:
    from saga.schedulers.online.environment import Environment


class OnlinePolicy(ABC):
    """The decision logic of an online algorithm: "if X is observed, do Y".

    A policy is consulted once per simulation step. It inspects the environment
    and either returns a revised :class:`Schedule` (the *Y*) or ``None`` to leave
    the current schedule unchanged. The condition (*X*) and the action (*Y*) live
    together in :meth:`update` because they are coupled in practice — an action
    needs to know exactly what its condition observed.

    Any per-run state (counters, EMA estimates, frontier heaps) is held as
    instance attributes and cleared in :meth:`reset`. A policy may compose other
    policies internally (e.g. Inspirit pins a priority task, then delegates the
    remaining slots to a fill policy).
    """

    def reset(self) -> None:
        """Clear per-run state. Called by :meth:`Environment.reset` before each run."""
        pass

    @abstractmethod
    def update(self, environment: "Environment") -> Optional[Schedule]:
        """Inspect the environment this step and optionally revise the schedule.

        Args:
            environment: The current environment state.

        Returns:
            A revised Schedule, or None to leave the current schedule unchanged.
        """
        raise NotImplementedError


# Concrete policies. Imported after OnlinePolicy is defined (they subclass it);
# re-exported for `from ...policy import X`.
from saga.schedulers.online.policy.reschedule import (  # noqa: E402
    ReschedulePolicy,
    ConditionalReschedulePolicy,
    RandomReschedulePolicy10,
    RandomReschedulePolicy25,
    RandomReschedulePolicy50,
)
from saga.schedulers.online.policy.inspirit import InspiritPolicy  # noqa: E402
from saga.schedulers.online.policy.frontier_fill import FrontierFillPolicy  # noqa: E402

__all__ = [
    "OnlinePolicy",
    "ReschedulePolicy",
    "ConditionalReschedulePolicy",
    "RandomReschedulePolicy10",
    "RandomReschedulePolicy25",
    "RandomReschedulePolicy50",
    "InspiritPolicy",
    "FrontierFillPolicy",
]
