from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from saga import Schedule, ScheduledTask

if TYPE_CHECKING:
    from saga.schedulers.online.environment import Environment


def _all_scheduled_tasks(environment: "Environment"):
    """Yield every ScheduledTask in the current schedule."""
    for tasks in environment.schedule.mapping.values():
        yield from tasks


class Trigger:
    """Base class for events that signal the controller to act.

    Subclass to carry event-specific data (e.g. which task completed,
    how far actual duration deviated from the estimate, etc.).
    """
    pass


class StepStrategy(ABC):
    """Defines what constitutes one step and when to advance the clock."""

    @abstractmethod
    def next_step(self, environment: "Environment") -> Optional[float]:
        """Return the next simulation time, or None when the simulation is complete.

        Args:
            environment: The current environment state.

        Returns:
            The timestamp to advance to, or None if there are no more steps.
        """
        raise NotImplementedError


class TaskCompletionStep(StepStrategy):
    """Advance the clock to the next task completion event.

    Returns None when all tasks have finished (no end time remains
    strictly after the current time).
    """
    def next_step(self, environment: "Environment") -> Optional[float]:
        finished = environment.get_next_task_finish(environment.schedule)
        return finished.end if finished is not None else None


class TaskStartStep(StepStrategy):
    """Advance the clock to the next task start event.

    Returns None when no task has a start time strictly after the
    current time (i.e. all tasks have already started).
    """

    def next_step(self, environment: "Environment") -> Optional[float]:
        finished = environment.get_next_task_start(environment.schedule)
        return finished.start if finished is not None else None


class TaskEventStep(StepStrategy):
    """Advance the clock to whichever comes first: a task start or a task completion.

    Combines TaskCompletionStep and TaskStartStep so every task boundary
    (start or end) is a step. Returns None when no future events remain.
    """

    def next_step(self, environment: "Environment") -> Optional[float]:
        next_finish = environment.get_next_task_finish()
        next_start = environment.get_next_task_start()
        candidates = [
            t for t in (
                next_finish.end if next_finish is not None else None,
                next_start.start if next_start is not None else None,
            )
            if t is not None
        ]
        return min(candidates) if candidates else None


class TimeStep(StepStrategy):
    """Advance the clock by a fixed interval each step.

    The simulation ends when no task has an end time strictly after the
    current time (i.e. the schedule has been fully executed).

    Args:
        dt: The fixed time increment per step.
    """

    def __init__(self, dt: float) -> None:
        self.dt = dt

    def next_step(self, environment: "Environment") -> Optional[float]:
        has_future_events = any(
            task.end > environment.current_time
            for task in _all_scheduled_tasks(environment)
        )
        if not has_future_events:
            return None
        return environment.current_time + self.dt


class Observer(ABC):
    """Stateful logic for *when* to fire — the X in "trigger on X, do Y".

    Inspects the environment at each step and returns a Trigger when its
    condition is met, or None to take no action.  Observers may be attached
    to an Environment with no Controller to passively watch state without
    ever modifying the schedule.
    """

    @abstractmethod
    def observe(self, environment: "Environment") -> Optional[Trigger]:
        """Inspect the environment and optionally emit a trigger.

        Args:
            environment: The current environment state.

        Returns:
            A Trigger if the condition is met, else None.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any internal state accumulated between steps.

        Called by Environment.reset() so a fresh simulation starts cleanly.
        Stateful observers (e.g. those that accumulate deltas) should override
        this; the default is a no-op.
        """
        pass


# ---------------------------------------------------------------------------
# Concrete triggers
# ---------------------------------------------------------------------------

class OnStepTrigger(Trigger):
    """Fired on every step.  Carries no extra data; acts as a type discriminator."""
    pass


class ReadyChangeTrigger(Trigger):
    """Fired when the cumulative absolute change in ready-task count reaches n.

    Attributes:
        ready_count: Number of ready tasks at the moment the trigger fired.
        cumulative_delta: The accumulated delta that crossed the threshold.
    """

    def __init__(self, ready_count: int, cumulative_delta: int) -> None:
        self.ready_count = ready_count
        self.cumulative_delta = cumulative_delta


# ---------------------------------------------------------------------------
# Concrete observers
# ---------------------------------------------------------------------------

class OnStepObserver(Observer):
    """Fires an OnStepTrigger unconditionally on every step."""

    def observe(self, environment: "Environment") -> Optional[Trigger]:
        return OnStepTrigger()


class ReadyChangeObserver(Observer):
    """Fires when the cumulative absolute change in ready-task count reaches n.

    Tracks |ready_now - ready_last| at each step and accumulates.  When the
    running sum reaches n, a ReadyChangeTrigger is emitted and the accumulator
    resets to zero.

    Example with n=5:
        step 0:  0 ready  →  delta=0,  cumulative=0
        step 1:  2 ready  →  delta=2,  cumulative=2
        step 2:  3 ready  →  delta=1,  cumulative=3
        step 3:  2 ready  →  delta=1,  cumulative=4
        step 4:  5 ready  →  delta=3,  cumulative=7  →  TRIGGER, reset to 0

    Args:
        delta_ready_tasks: Cumulative-change threshold that fires the trigger. Recommended to set the proportional to the number of workers
    """

    def __init__(self, delta_ready_tasks: int) -> None:
        self.delta_ready_tasks = delta_ready_tasks
        self._last_ready_count: int = 0
        self._cumulative_delta: int = 0

    def reset(self) -> None:
        self._last_ready_count = 0
        self._cumulative_delta = 0

    def observe(self, environment: "Environment") -> Optional[Trigger]:
        current_ready = len(environment.ready_tasks)
        self._cumulative_delta += current_ready - self._last_ready_count
        self._last_ready_count = current_ready

        if abs(self._cumulative_delta) >= self.delta_ready_tasks:
            fired_delta = self._cumulative_delta
            self._cumulative_delta = 0
            return ReadyChangeTrigger(
                ready_count=current_ready,
                cumulative_delta=fired_delta,
            )

        return None


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class Controller(ABC):
    """The Y in "trigger on X, do Y" — responds to a trigger by modifying the schedule.

    A Controller may internally wrap or inherit from a Scheduler to produce a
    revised schedule (e.g. re-scheduling remaining tasks around committed ones).
    """

    @abstractmethod
    def control(self, environment: "Environment", trigger: Trigger) -> Schedule:
        """Produce a modified schedule in response to the trigger.

        Args:
            environment: The current environment state.
            trigger: The trigger emitted by the observer.  Use isinstance()
                     to branch on trigger type if the controller handles more
                     than one.

        Returns:
            An updated Schedule reflecting the controller's changes.
        """
        raise NotImplementedError
