import heapq
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set, cast
import numpy as np

from saga import Network, Schedule, ScheduledTask, Scheduler, TaskGraph, TaskGraphEdge, TaskGraphNode, NetworkEdge, NetworkNode
from saga.schedulers.online.components import Controller, Observer, StepStrategy, TaskEventStep, OnStepObserver
from saga.schedulers.stochastic.estimate_stochastic_scheduler import EstimateStochasticScheduler
from saga.stochastic import StochasticNetwork, StochasticScheduler, StochasticSchedule, StochasticScheduledTask, StochasticTaskGraph
from saga.schedulers.online.environment import Environment
from saga.schedulers.parametric.components import GreedyInsert, GreedyInsertCompareFuncs
from saga.utils.random_variable import RandomVariable

logger = logging.getLogger(__name__)


class StochasticEnvironment(Environment):
    """Environment wrapper that adds stochasticity to task execution and network transmission times.

    The base Environment tracks the deterministic "nominal" schedule based on the input
    Network and TaskGraph. The StochasticEnvironment maintains a separate "actual" schedule
    that reflects the realized start/end times of tasks and transmissions under stochasticity.
    The step strategy and observer/controller can read from either schedule as needed.
    *** WIP ***
    
    """
    def __init__(
        self,
        network: StochasticNetwork,
        task_graph: StochasticTaskGraph,
        step_strategy: StepStrategy,
        observer: Observer,
        scheduler: Scheduler,
        estimate: Callable[[RandomVariable], float],
        controller: Optional[Controller] = None,
        on_step: Optional[Callable[["Environment"], None]] = None,
        seed: Optional[int] = None
    ) -> None:
        super().__init__(
            network=cast(Network, network),
            task_graph=cast(TaskGraph, task_graph),
            step_strategy=step_strategy,
            observer=observer,
            scheduler=scheduler,
            controller=controller,
            on_step=on_step
        )
        self._stochatic_task_graph = task_graph
        self._stochastic_network = network
        self.seed = seed
        self.actual_task_graph = task_graph.sample()
        self.actual_network = network.sample()
        self._estimate = estimate
        self.stochastic_scheduler = EstimateStochasticScheduler(
            scheduler=scheduler,
            estimate=estimate
        )

        self.initial_estimate_schedule, self.network, self.task_graph = self.stochastic_scheduler.schedule(network=network,task_graph=task_graph)
        self.estimate_schedule: StochasticSchedule = self.initial_estimate_schedule
        self.schedule_actual = self.estimate_schedule.determinize(self.actual_network, self.actual_task_graph)

    def reset(self) -> None:
        super().reset()
        if self.seed is not None:
            np.random.seed(self.seed)
        self.actual_task_graph = self._stochatic_task_graph.sample()
        self.actual_network = self._stochastic_network.sample()
        self.initial_estimate_schedule, self.network, self.task_graph = self.stochastic_scheduler.schedule(
            network=self._stochastic_network, task_graph=self._stochatic_task_graph
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


class FrontierEnvironment(Environment):
    """Environment for frontier-based online scheduling. Algorithms include FIFO, LIFO, FrontierHEFT, etc."""

    def __init__(
        self,
        network: Network,
        task_graph: TaskGraph,
        scheduler: Optional[Scheduler] = None,
        step_strategy: Optional[StepStrategy] = None,
        observer: Optional[Observer] = None,
        controller: Optional[Controller] = None,
        on_step: Optional[Callable[["Environment"], None]] = None,
    ) -> None:
        super().__init__(
            network=network,
            task_graph=task_graph,
            scheduler=scheduler,
            step_strategy=step_strategy if step_strategy is not None else TaskEventStep(),
            observer=observer if observer is not None else OnStepObserver(),
            controller=controller,
            on_step=on_step,
        )
        self.ready_condition: str = "p_complete"
        self.ready_node_only: bool = True
        self.frontier: List[Tuple[float, str]] = []
        self.frontier_set: Set[str] = set()
        self._bootstrap_insert = GreedyInsert(
            append_only=False,
            compare=GreedyInsertCompareFuncs.EST,
            critical_path=False,
        )
        # Returns a sort key for the frontier heap; usually a float, but subclasses
        # (e.g. FrontierHeftEnvironment) may return a tuple for lexicographic ordering.
        self.priority_condition: Callable[[TaskGraphNode], Any] = lambda _: self.current_time

    def reset(
        self,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> None:
        """Initialize state, optionally seeding from a partial schedule."""
        self.current_time = min_start_time
        self.prev_nready = 0
        self.finished_tasks = set()
        self.running_tasks = set()
        self.ready_tasks = set()
        self.unready_tasks = set()
        self.available_nodes = set()
        self.occupied_nodes = set()
        self._step = 0
        self.history = []
        self.observer.reset()
        if self.controller is not None:
            self.controller.reset()
        self.frontier = []
        self.frontier_set = set()
        if schedule is not None:
            self.schedule = schedule
        else:
            self.schedule = Schedule(task_graph=self.task_graph, network=self.network)
            for task in self.task_graph.tasks:
                if self.task_graph.in_degree(task) == 0:
                    self._bootstrap_insert.call(
                        self.network, self.task_graph, self.schedule, task.name, min_start_time=min_start_time
                    )
        self._update_task_state()
        self._update_network_state()

    def _update_task_state(self) -> None:
        self.prev_nready = len(self.ready_tasks)

        self.finished_tasks = self.get_finished_tasks()
        self.running_tasks = self.get_running_tasks()
        self.committed = {t for tasks in self.schedule.mapping.values() for t in tasks}

        finished_names = {t.name for t in self.finished_tasks}
        committed_names = {t.name for t in self.committed}
        finished_or_running = {t.name for t in self.running_tasks} | finished_names
        stale = committed_names & self.frontier_set
        if stale:
            self.frontier_set -= stale
            self.frontier = [(t, n) for t, n in self.frontier if n not in stale]
            heapq.heapify(self.frontier)

        for task in self.task_graph.tasks:
            name = task.name
            if name in committed_names or name in self.frontier_set:
                continue
            predecessors = {dep.source for dep in self.task_graph.in_edges(name)}
            if self.ready_condition == "p_complete":
                if predecessors.issubset(finished_names):
                    self.frontier_set.add(name)
                    self.unready_tasks.discard(task)
                    heapq.heappush(self.frontier, (self.priority_condition(task), name))
            elif self.ready_condition == "p_committed":
                if predecessors.issubset(finished_or_running):
                    self.frontier_set.add(name)
                    self.unready_tasks.discard(task)
                    heapq.heappush(self.frontier, (self.priority_condition(task), name))
            elif self.ready_condition == "p_scheduled":
                if predecessors.issubset(committed_names):
                    self.frontier_set.add(name)
                    self.unready_tasks.discard(task)
                    heapq.heappush(self.frontier, (self.priority_condition(task), name))

        task_by_name = {task.name: task for task in self.task_graph.tasks}
        self.ready_tasks = {task_by_name[name] for name in self.frontier_set}
