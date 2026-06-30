import logging
from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, List, Optional, Set

from saga import Network, NetworkNode, TaskGraph, Schedule, Scheduler, ScheduledTask
from saga.schedulers.online.components import Controller, Observer, StepStrategy, Trigger

logger = logging.getLogger(__name__)


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

    Holds the network, task graph, and current schedule state, and runs
    a step -> observe -> control loop until the simulation is complete.

    Typical usage::

        env = Environment(network, task_graph, scheduler, step_strategy, observer, controller)
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
        step_strategy: StepStrategy,
        observer: Observer,
        scheduler: Optional[Scheduler] = None,
        controller: Optional[Controller] = None,
        on_step: Optional[Callable[["Environment"], None]] = None,
    ) -> None:
        """
        Args:
            network:       The compute network tasks run on.
            task_graph:    The DAG of tasks to schedule.
            scheduler:     Produces the initial schedule and is the default for RescheduleController.
            step_strategy: Determines when the clock advances (e.g. task completion, fixed interval).
            observer:      Watches state each step and emits a Trigger when its condition is met.
            controller:    Optional. Acts on a Trigger to modify the schedule. If None, the
                           observer still runs every step but nothing changes the schedule.
            on_step:       Optional callback invoked at the end of every step, after state has been
                           updated and any controller action has been applied. Receives the
                           Environment instance so it has full access to current_time, schedule,
                           finished_tasks, ready_tasks, history, etc. Useful for logging, recording
                           custom metrics, or any side effect that should happen every step without
                           subclassing Environment. Examples::

                               # Print progress each step
                               Environment(..., on_step=lambda e: print(f"t={e.current_time:.2f}"))

                               # Accumulate a custom metric alongside the built-in history
                               energy_log = []
                               Environment(..., on_step=lambda e: energy_log.append(measure(e)))
        """
        self.network = network
        self.task_graph = task_graph
        self.scheduler = scheduler
        self.step_strategy = step_strategy
        self.observer = observer
        self.controller = controller
        self.on_step = on_step

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
        # Seeded with TaskGraphNodes but discarded against ScheduledTasks in the base
        # env (and TaskGraphNodes in FrontierEnvironment); only `.name` is relied on.
        self.unready_tasks: Set[Any] = set(self.task_graph.tasks)
        self.available_nodes: Set[NetworkNode] = set()
        self.occupied_nodes: Set[NetworkNode] = set()
        self._step: int = 0
        self.history: List[StepRecord] = []
        self.prev_nready: int = 0

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
        self.unready_tasks = set(self.task_graph.tasks)
        self.available_nodes = set()
        self.occupied_nodes = set()
        self._step = 0
        self.history = []
        self.prev_nready = 0
        self.observer.reset()
        if self.controller is not None:
            self.controller.reset()
        if self.scheduler is None:
            raise ValueError("Environment.reset() requires a scheduler to produce the initial schedule.")
        self.schedule = self.scheduler.schedule(self.network, self.task_graph)
        self._update_task_state()
        self._update_network_state()

    def step(self) -> bool:
        """Advance the simulation by one step.

        Asks the StepStrategy for the next time point, advances the clock,
        refreshes task state, asks the Observer whether to act, and if so
        calls the Controller to update the schedule.

        Returns:
            True if the simulation should continue, False if it is complete.
        """
        next_time = self.step_strategy.next_step(self)
        if next_time is None:
            return False

        self.current_time = next_time
        self._update_task_state()
        self._update_network_state()

        if self.controller is not None:
            self.controller.pre_step(self)

        trigger: Optional[Trigger] = self.observer.observe(self)
        if trigger is not None and self.controller is not None:
            self.schedule = self.controller.control(self, trigger)
            self._update_task_state()
            self._update_network_state()

        self.history.append(StepRecord(
            step=self._step,
            time=self.current_time,
            finished_tasks=frozenset(t.name for t in self.finished_tasks),
            running_tasks=frozenset(t.name for t in self.running_tasks),
            ready_tasks=frozenset(t.name for t in self.ready_tasks),
            unready_tasks=frozenset(t.name for t in self.unready_tasks),
            makespan=self.schedule.makespan,
        ))
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

    def get_next_task_finish(self, schedule: Optional[Schedule] = None) -> Optional[ScheduledTask]:
        """Return the task with the smallest end time strictly after current_time."""
        schedule = schedule or self.schedule
        candidates = [
            task
            for tasks in schedule.mapping.values()
            for task in tasks
            if task.end > self.current_time
        ]
        return min(candidates, key=lambda t: t.end, default=None)

    def get_next_task_start(self, schedule: Optional[Schedule] = None) -> Optional[ScheduledTask]:
        """Return the task with the smallest start time strictly after current_time."""
        schedule = schedule or self.schedule
        candidates = [
            task
            for tasks in schedule.mapping.values()
            for task in tasks
            if task.start > self.current_time
        ]
        return min(candidates, key=lambda t: t.start, default=None)

    def get_running_tasks(self, schedule: Optional[Schedule] = None) -> Set[ScheduledTask]:
        """Return tasks that have started but not yet finished at current_time."""
        schedule = schedule or self.schedule
        return {
            task
            for tasks in schedule.mapping.values()
            for task in tasks
            if task.start <= self.current_time < task.end
        }

    def get_finished_tasks(self, schedule: Optional[Schedule] = None) -> Set[ScheduledTask]:
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
                predecessors = {dep.source for dep in self.task_graph.in_edges(task.name)}
                if predecessors.issubset(finished_names):
                    self.ready_tasks.add(task)
                    self.unready_tasks.discard(task)

    def _update_network_state(self) -> None:
        """Recompute available and occupied node sets from the current running tasks."""
        occupied_node_names = {task.node for task in self.running_tasks}
        self.occupied_nodes = {node for node in self.network.nodes if node.name in occupied_node_names}
        self.available_nodes = {node for node in self.network.nodes if node.name not in occupied_node_names}


