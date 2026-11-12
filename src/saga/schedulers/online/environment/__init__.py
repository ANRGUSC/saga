"""Online scheduling simulation core: the Environment loop, step functions, and StepRecord.

Concrete environment variants live alongside this module and are re-exported here:
:class:`FrontierEnvironment` (frontier.py) and :class:`StochasticEnvironment` (stochastic.py).
The pluggable decision logic (:class:`OnlinePolicy` and its implementations) lives in the
sibling ``policy`` package; this package never imports it at runtime, keeping the
dependency one-directional (policy -> environment).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, TYPE_CHECKING

from saga import Network, NetworkNode, TaskGraph, Schedule, Scheduler, ScheduledTask

if TYPE_CHECKING:
    from saga.schedulers.online.policy import OnlinePolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step functions — decide when the simulation clock advances.
#
# A step function maps the environment to the next simulation time, or None
# when the simulation is complete. This is a plain callable rather than a class
# hierarchy: the common case (advance to the next task completion) is the
# default, and the rare ones (event boundaries, fixed intervals) are just other
# functions you can pass to Environment(step=...).
# ---------------------------------------------------------------------------

StepFunction = Callable[["Environment"], Optional[float]]


def next_completion(environment: "Environment") -> Optional[float]:
    """Advance to the next task completion, or None when all tasks have finished."""
    finished = environment.get_next_task_finish()
    return finished.end if finished is not None else None


def next_start(environment: "Environment") -> Optional[float]:
    """Advance to the next task start, or None when every task has started."""
    started = environment.get_next_task_start()
    return started.start if started is not None else None


def next_event(environment: "Environment") -> Optional[float]:
    """Advance to whichever comes first: a task start or a task completion.

    Every task boundary (start or end) is a step. Returns None when no future
    events remain.
    """
    next_finish = environment.get_next_task_finish()
    next_start_task = environment.get_next_task_start()
    candidates = [
        t
        for t in (
            next_finish.end if next_finish is not None else None,
            next_start_task.start if next_start_task is not None else None,
        )
        if t is not None
    ]
    return min(candidates) if candidates else None


def time_step(dt: float) -> StepFunction:
    """Build a step function that advances the clock by a fixed interval ``dt``.

    The simulation ends when no task has an end time strictly after the current
    time (i.e. the schedule has been fully executed).
    """

    def step(environment: "Environment") -> Optional[float]:
        has_future_events = any(
            task.end > environment.current_time
            for tasks in environment.schedule.mapping.values()
            for task in tasks
        )
        if not has_future_events:
            return None
        return environment.current_time + dt

    return step


@dataclass
class StepRecord:
    """Snapshot of environment state captured at the end of one step.

    Accessible via Environment.history after run() or during on_step callbacks.

    Attributes:
        step:          Zero-based step index.
        time:          Simulation clock at this step (time the step landed on).
        finished_tasks: Names of tasks that had completed by this time.
        running_tasks:  Names of tasks executing at this time.
        ready_tasks:    Names of tasks whose predecessors are all done but have not started.
        unready_tasks:  Names of tasks still blocked by unfinished predecessors.
        makespan:       Current schedule makespan at this step.
    """

    step: int
    time: float
    finished_tasks: FrozenSet[str]
    running_tasks: FrozenSet[str]
    ready_tasks: FrozenSet[str]
    unready_tasks: FrozenSet[str]
    makespan: float


class Environment:
    """Orchestrates the online scheduling simulation loop.

    Holds the network, task graph, and current schedule state, and runs a
    step -> update loop until the simulation is complete. Each step advances the
    clock (via the step function), refreshes task/network state, and consults the
    policy, which may revise the remaining schedule.

    Typical usage::

        env = Environment(network, task_graph, scheduler=heft, policy=ReschedulePolicy())
        final_schedule = env.run()

    Or step-by-step for inspection::

        env.reset()
        while env.step():
            print(env.current_time, env.schedule.makespan)
    """

    def __init__(
        self,
        network: Network,
        task_graph: TaskGraph,
        scheduler: Optional[Scheduler] = None,
        policy: Optional["OnlinePolicy"] = None,
        step: StepFunction = next_completion,
        on_step: Optional[Callable[["Environment"], None]] = None,
        node_constraints: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        """
        Args:
            network:    The compute network tasks run on.
            task_graph: The DAG of tasks to schedule.
            scheduler:  Produces the initial schedule and is used by policies that
                        reschedule remaining tasks (e.g. ReschedulePolicy).
            policy:     Optional. Consulted every step to revise the schedule. If
                        None, the simulation just plays the initial schedule forward.
            step:       Function deciding when the clock advances. Defaults to
                        next_completion (advance to the next task completion).
            on_step:    Optional callback invoked at the end of every step, after
                        state has been updated and any policy action applied.
                        Receives the Environment so it can read current_time,
                        schedule, ready_tasks, history, etc. Useful for logging or
                        recording custom metrics without subclassing. Examples::

                            # Print progress each step
                            Environment(..., on_step=lambda e: print(f"t={e.current_time:.2f}"))

                            # Accumulate a custom metric alongside the built-in history
                            energy_log = []
                            Environment(..., on_step=lambda e: energy_log.append(measure(e)))
        """
        self.network = network
        self.task_graph = task_graph
        self.scheduler = scheduler
        self.policy = policy
        self.step_fn = step
        self.on_step = on_step
        # Per-instance placement constraints, threaded into every schedule built during
        # the run (the initial schedule and every policy re-plan via build_partial_schedule).
        self.node_constraints = node_constraints

        # Mutable simulation state — initialized by reset()
        self.current_time: float = 0.0
        # Produced by reset() before any step runs; typed non-optional since all
        # public entry points (run/step) call reset() first.
        self.schedule: Schedule = None  # type: ignore[assignment]
        self.finished_tasks: Set[ScheduledTask] = set()
        self.running_tasks: Set[ScheduledTask] = set()
        self.committed: Set[ScheduledTask] = set()
        # Holds task-like objects (ScheduledTask in batch environments, TaskGraphNode
        # in FrontierEnvironment); consumers only rely on the `.name` attribute.
        self.ready_tasks: Set[Any] = set()
        self.unready_tasks: Set[str] = {task.name for task in self.task_graph.tasks}
        self.available_nodes: Set[NetworkNode] = set()
        self.occupied_nodes: Set[NetworkNode] = set()
        self._step: int = 0
        self.history: List[StepRecord] = []
        self.prev_nready: int = 0
        self.reschedule_count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Initialize state and produce the first schedule via the base scheduler."""
        self.current_time = 0.0
        self.finished_tasks = set()
        self.running_tasks = set()
        self.ready_tasks = set()
        self.committed = set()
        self.unready_tasks = {task.name for task in self.task_graph.tasks}
        self.available_nodes = set()
        self.occupied_nodes = set()
        self._step = 0
        self.history = []
        self.prev_nready = 0
        self.reschedule_count = 0
        if self.policy is not None:
            self.policy.reset()
        if self.scheduler is None:
            raise ValueError(
                "Environment.reset() requires a scheduler to produce the initial schedule."
            )
        # Only forward node_constraints when set, so plain schedulers (whose schedule()
        # does not take the argument) still work in the unconstrained case.
        kwargs: Dict[str, Any] = (
            {"node_constraints": self.node_constraints} if self.node_constraints else {}
        )
        self.schedule = self.scheduler.schedule(self.network, self.task_graph, **kwargs)
        self._update_task_state()
        self._update_network_state()

    def step(self) -> bool:
        """Advance the simulation by one step.

        Asks the step function for the next time point, advances the clock,
        refreshes task state, and consults the policy. If the policy returns a
        revised schedule, it replaces the current one.

        Returns:
            True if the simulation should continue, False if it is complete.
        """
        next_time = self.step_fn(self)
        if next_time is None:
            return False

        self.current_time = next_time
        self._update_task_state()
        self._update_network_state()

        if self.policy is not None:
            revised = self.policy.update(self)
            if revised is not None:
                self.schedule = revised
                self._update_task_state()
                self._update_network_state()

        self.history.append(
            StepRecord(
                step=self._step,
                time=self.current_time,
                finished_tasks=frozenset(t.name for t in self.finished_tasks),
                running_tasks=frozenset(t.name for t in self.running_tasks),
                ready_tasks=frozenset(t.name for t in self.ready_tasks),
                unready_tasks=frozenset(self.unready_tasks),
                makespan=self.schedule.makespan,
            )
        )
        self._step += 1

        if self.on_step is not None:
            self.on_step(self)

        return True

    def run(self) -> Schedule:
        """Run the full simulation loop and return the final schedule.

        Returns:
            The Schedule produced after all steps have been taken.
        """
        self.reset()
        while self.step():
            pass
        return self.schedule

    # ------------------------------------------------------------------
    # Schedule state queries
    # ------------------------------------------------------------------

    def get_next_task_finish(
        self, schedule: Optional[Schedule] = None
    ) -> Optional[ScheduledTask]:
        """Return the task with the smallest end time strictly after current_time."""
        schedule = schedule or self.schedule
        candidates = [
            task
            for tasks in schedule.mapping.values()
            for task in tasks
            if task.end > self.current_time
        ]
        return min(candidates, key=lambda t: t.end, default=None)

    def get_next_task_start(
        self, schedule: Optional[Schedule] = None
    ) -> Optional[ScheduledTask]:
        """Return the task with the smallest start time strictly after current_time."""
        schedule = schedule or self.schedule
        candidates = [
            task
            for tasks in schedule.mapping.values()
            for task in tasks
            if task.start > self.current_time
        ]
        return min(candidates, key=lambda t: t.start, default=None)

    def get_running_tasks(
        self, schedule: Optional[Schedule] = None
    ) -> Set[ScheduledTask]:
        """Return tasks that have started but not yet finished at current_time."""
        schedule = schedule or self.schedule
        return {
            task
            for tasks in schedule.mapping.values()
            for task in tasks
            if task.start <= self.current_time < task.end
        }

    def get_finished_tasks(
        self, schedule: Optional[Schedule] = None
    ) -> Set[ScheduledTask]:
        """Return tasks whose end time is at or before current_time."""
        schedule = schedule or self.schedule
        return {
            task
            for tasks in schedule.mapping.values()
            for task in tasks
            if task.end <= self.current_time
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_task_state(self) -> None:
        """Recompute all four task-state sets from the current schedule and task graph."""
        self.prev_nready = len(self.ready_tasks)
        self.finished_tasks = self.get_finished_tasks()
        self.running_tasks = self.get_running_tasks()

        finished_names = {t.name for t in self.finished_tasks}
        committed_names = finished_names | {t.name for t in self.running_tasks}

        self.ready_tasks = set()
        for tasks in self.schedule.mapping.values():
            for task in tasks:
                if task.name in committed_names:
                    continue
                predecessors = {
                    dep.source for dep in self.task_graph.in_edges(task.name)
                }
                if predecessors.issubset(finished_names):
                    self.ready_tasks.add(task)

        ready_names = {task.name for task in self.ready_tasks}
        self.unready_tasks = {
            task.name
            for task in self.task_graph.tasks
            if task.name not in committed_names and task.name not in ready_names
        }

    def _update_network_state(self) -> None:
        """Recompute available and occupied node sets from the current running tasks."""
        occupied_node_names = {task.node for task in self.running_tasks}
        self.occupied_nodes = {
            node for node in self.network.nodes if node.name in occupied_node_names
        }
        self.available_nodes = {
            node for node in self.network.nodes if node.name not in occupied_node_names
        }


# Concrete environment variants. Imported at the bottom (after Environment is
# defined) so they can subclass it; re-exported for `from ...environment import X`.
from saga.schedulers.online.environment.frontier import FrontierEnvironment  # noqa: E402
from saga.schedulers.online.environment.stochastic import StochasticEnvironment  # noqa: E402

__all__ = [
    "Environment",
    "FrontierEnvironment",
    "StochasticEnvironment",
    "StepRecord",
    "StepFunction",
    "next_completion",
    "next_start",
    "next_event",
    "time_step",
]
