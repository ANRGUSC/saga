import heapq
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union, Set
import numpy as np

from saga import Network, Schedule, ScheduledTask, Scheduler, TaskGraph, TaskGraphEdge, TaskGraphNode, NetworkEdge, NetworkNode
from saga.schedulers.online.components import Controller, Observer, StepStrategy, TaskEventStep, OnStepObserver
from saga.schedulers.stochastic.estimate_stochastic_scheduler import EstimateStochasticScheduler
from saga.stochastic import StochasticNetwork, StochasticScheduler, StochasticSchedule, StochasticScheduledTask, StochasticTaskGraph
from saga.schedulers.online.environment import Environment
from saga.schedulers.throughput.inspirit import (
    compute_inspiring_ability,
    compute_inspiring_effeciency,
)
from saga.schedulers.parametric.components import GreedyInsert, GreedyInsertCompareFuncs
from saga.utils.random_variable import RandomVariable

logger = logging.getLogger(__name__)


class InspiritEnvironment(Environment):
    """Environment for the Inspirit throughput-maximisation algorithm.

    Extends the base Environment with two always-current priority frontiers
    over the ready-task set:
      - efficiency_frontier  ranked by inspiring efficiency (tasks that unlock
                             the most work within a time window)
      - ability_frontier     ranked by inspiring ability (total downstream tasks)

    Both heaps are rebuilt from scratch every time _update_task_state() runs,
    so they are always a perfect mirror of the current ready_tasks set.
    O(n) rebuild cost (heapify) is negligible for typical task graph sizes.

    Also tracks the prev_nready, peak, and cur_state bookkeeping that
    InspiritController reads to decide which frontier to pop from.
    """

    INC = "INC"
    DEC = "DEC"

    def __init__(
        self,
        network: Network,
        task_graph: TaskGraph,
        step_strategy: StepStrategy,
        observer: Observer,
        scheduler: Scheduler,
        time_window:Optional[float] = None,
        controller: Optional[Controller] = None,
        on_step: Optional[Callable[["Environment"], None]] = None,
        dec_step: Optional[int] = None,
        s_inc: Optional[int] = None,
        s_dec: Optional[int] = None,
    ) -> None:
        # Call super first so self.network / self.task_graph are set before
        # we compute the static scores that depend on them.
        super().__init__(
            network=network,
            task_graph=task_graph,
            step_strategy=step_strategy,
            observer=observer,
            scheduler=scheduler,
            controller=controller,
            on_step=on_step,
        )
        
        self.time_window = len(self.network.nodes) * (np.mean([task.cost for task in task_graph.tasks])/np.mean([node.speed for node in network.nodes])) if time_window is None else time_window
        #print ("Network nodes count: ", len(self.network.nodes), "Mean task cost: ", np.mean([task.cost for task in task_graph.tasks]), "Mean node speed: ", np.mean([node.speed for node in network.nodes]))
        #print(f"Using time_window={self.time_window:.2f} for efficiency score computation.")
        # Static scores — computed once from the task graph structure.
        # These never change between steps; only the heap membership changes.
        self.efficiency_ranks: Dict[str, float] = compute_inspiring_effeciency(
            task_graph, network, time_window=self.time_window
        )
        self.ability_ranks: Dict[str, float] = compute_inspiring_ability(task_graph)

        # Priority frontiers over ready_tasks.
        # Format: min-heap of (-score, task_name) so largest score is at index 0.
        # Rebuilt every _update_task_state() call — always in sync, no lazy deletion.
        self.efficiency_frontier: List[Tuple[float, str]] = []
        self.ability_frontier: List[Tuple[float, str]] = []

        # Cross-trigger bookkeeping read by InspiritController.
        # dec_step, s_inc, s_dec default to number of workers when not specified.
        workers = len(network.nodes)
        self._dec_step_override = dec_step
        self._s_inc_override = s_inc
        self._s_dec_override = s_dec

        self.prev_nready: int = 0
        self.peak: int = 0
        self.cur_state: Optional[str] = None
        self.dec_step: int = dec_step if dec_step is not None else workers
        self.k_inc: float = 0.0
        self.cur_k: float = 0.0
        self.s_inc: int = s_inc if s_inc is not None else workers
        self.s_dec: int = s_dec if s_dec is not None else workers
        self.s_dec_count: int = 0
        self.c = max(1, workers // 2)  # constant factor for hysteresis in decreasing phase, not defined in paper
        self.last_dispatched: Optional[str] = None
        self.last_dispatch_type: Optional[str] = None  # "efficiency" or "ability"
        # Reference point for k rate computation — updated only on each k update,
        # so rate is measured over the inter-dispatch interval, not step-to-step.
        self._last_k_time: float = 0.0
        self._last_k_nready: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        # Recompute static scores in case network/task_graph changed between runs
        self.efficiency_ranks = compute_inspiring_effeciency(
            self.task_graph, self.network, time_window=self.time_window
        )
        self.ability_ranks = compute_inspiring_ability(self.task_graph)

        # Frontiers are rebuilt by _update_task_state() which super().reset() calls
        self.efficiency_frontier = []
        self.ability_frontier = []

        # Reset bookkeeping
        workers = len(self.network.nodes)
        self.prev_nready = 0
        self.peak = 0
        self.cur_state = None
        self.dec_step = self._dec_step_override if self._dec_step_override is not None else workers
        self.k_inc = 0.0
        self.cur_k = 0.0
        self.s_inc = self._s_inc_override if self._s_inc_override is not None else workers
        self.s_dec = self._s_dec_override if self._s_dec_override is not None else workers
        self.s_dec_count = 0
        self.c = max(1, workers // 2)
        self.last_dispatched = None
        self.last_dispatch_type = None
        self._last_k_time = 0.0
        self._last_k_nready = 0

    # ------------------------------------------------------------------
    # State update — called every step before observe/control
    # ------------------------------------------------------------------

    def _update_task_state(self) -> None:
        # Store ready count from the previous step before overwriting
        self.prev_nready = len(self.ready_tasks)

        # Reset per-step dispatch record so on_step only sees a name if
        # control() actually fired and chose a task this step.
        self.last_dispatched = None
        self.last_dispatch_type = None

        # Delegate base state computation (finished/running/ready/unready)
        super()._update_task_state()

        # Rebuild frontiers as a fresh heap over the current ready_tasks.
        ready_names = {t.name for t in self.ready_tasks}
        self.efficiency_frontier = [
            (-self.efficiency_ranks[name], name) for name in ready_names
        ]
        self.ability_frontier = [
            (-self.ability_ranks[name], name) for name in ready_names
        ]
        heapq.heapify(self.efficiency_frontier)
        heapq.heapify(self.ability_frontier)

    # ------------------------------------------------------------------
    # Frontier accessors used by InspiritController
    # ------------------------------------------------------------------

    def pop_highest_efficiency(self) -> Optional[str]:
        """Remove and return the ready task with the highest efficiency score."""
        if self.efficiency_frontier:
            _, name = heapq.heappop(self.efficiency_frontier)
            return name
        return None

    def pop_highest_ability(self) -> Optional[str]:
        """Remove and return the ready task with the highest ability score."""
        if self.ability_frontier:
            _, name = heapq.heappop(self.ability_frontier)
            return name
        return None
   

class StochasticEnvironment(Environment):
    """Environment wrapper that adds stochasticity to task execution and network transmission times.

    The base Environment tracks the deterministic "nominal" schedule based on the input
    Network and TaskGraph. The StochasticEnvironment maintains a separate "actual" schedule
    that reflects the realized start/end times of tasks and transmissions under stochasticity.
    The step strategy and observer/controller can read from either schedule as needed.
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
            network=network,
            task_graph=task_graph,
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
        # self.schedule_actual:Schedule = Schedule(network=self.actual_network, task_graph=self.actual_task_graph, mapping=None)
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
                    self.unready_tasks.discard()

        

class FrontierEnvironment(Environment):
    """Environment for simple frontier-based online scheduling. Algorithms include FIFO, LIFO, etc."""

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
        # Default insertion strategy used for bootstrapping root tasks in reset()
        self._bootstrap_insert = GreedyInsert(
            append_only=False,
            compare=GreedyInsertCompareFuncs.EST,
            critical_path=False,
        )
        self.priority_condition: Callable[[TaskGraphNode],float] = lambda _: self.current_time

    def reset(
        self,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> None:
        """Initialize state, optionally seeding from a partial schedule.

        When schedule is provided (e.g. from InspiritController), the committed
        tasks it contains are treated as already done and the simulation fills
        in the remaining tasks around them starting at min_start_time.
        When schedule is None, root tasks are bootstrapped so the simulation
        has events to advance to.
        """
        self.current_time = min_start_time
        self.finished_tasks = set()
        self.running_tasks = set()
        self.ready_tasks = set()
        self.unready_tasks = set()
        self.available_nodes = set()
        self.occupied_nodes = set()
        self._step = 0
        self.history = []
        self.observer.reset()
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
        self._update_task_state(self.pri)
        self._update_network_state()

    def _update_task_state(self, priority_definition:Callable[[TaskGraphNode],float]) -> None:
        if priority_definition is None:
            priority_definition = lambda _: self.current_time
        
        self.finished_tasks = self.get_finished_tasks()
        self.running_tasks = self.get_running_tasks()
        self.committed = {t for tasks in self.schedule.mapping.values() for t in tasks}

        finished_names = {t.name for t in self.finished_tasks}
        committed_names = {t.name for t in self.committed}
        finished_or_running = {t.name for t in self.running_tasks} | finished_names
        self.ready_tasks = set()
        stale = committed_names & self.frontier_set
        if stale:
            self.frontier_set -= stale
            self.frontier = [(t, n) for t, n in self.frontier if n not in stale]
            heapq.heapify(self.frontier)

        # Enqueue unscheduled tasks whose predecessors have all finished
        for task in self.task_graph.tasks:
            name = task.name
            #if name in scheduled_names or name in self.frontier_set:
            if name in committed_names or name in self.frontier_set:
                continue
            predecessors = {dep.source for dep in self.task_graph.in_edges(name)}
            #tasks are considered ready to schedule when all parents have completed
            if self.ready_condition == "p_complete":
                if predecessors.issubset(finished_names):
                    self.frontier_set.add(name)
                    self.ready_tasks.add(task)
                    self.unready_tasks.discard(task)
                    heapq.heappush(self.frontier, (priority_definition(task), name))
            #tasks are considered ready to schedule when all parents have completed or are running (committed)
            elif self.ready_condition == "p_committed":
                if predecessors.issubset(finished_or_running):
                    self.frontier_set.add(name)
                    self.ready_tasks.add(task)
                    self.unready_tasks.discard(task)
                    heapq.heappush(self.frontier, (priority_definition(task), name))
            #tasks are considered ready to schedule when all parents have been scheduled 
            elif self.ready_condition == "p_scheduled":
                if predecessors.issubset(committed_names):
                    self.frontier_set.add(name)
                    self.ready_tasks.add(task)
                    self.unready_tasks.discard(task)
                    heapq.heappush(self.frontier, (priority_definition(task), name))


