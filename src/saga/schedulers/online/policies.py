import logging
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import heapq
import numpy as np

from saga import Schedule, ScheduledTask
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.online.environment import OnlinePolicy
from saga.schedulers.online.environments import StochasticEnvironment, FrontierEnvironment
from saga.schedulers.parametric.components import GreedyInsert, GreedyInsertCompareFuncs
from saga.schedulers.throughput.inspirit import (
    compute_inspiring_ability,
    compute_inspiring_effeciency,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from saga.schedulers.online.environment import Environment


def _build_partial_schedule(environment: "Environment") -> Schedule:
    """Return a Schedule containing only the committed (finished + running) tasks."""
    partial = Schedule(environment.task_graph, environment.network)
    if isinstance(environment, StochasticEnvironment):
        environment.schedule_actual = environment.estimate_schedule.determinize(
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
        partial = _build_partial_schedule(environment)
        if isinstance(environment, StochasticEnvironment):
            new_estimate = environment.stochastic_scheduler.schedule(
                environment._stochastic_network,
                environment._stochastic_task_graph,
                schedule=partial,
                min_start_time=environment.current_time,
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


class InspiritPolicy(OnlinePolicy):
    """Inspirit throughput-maximisation policy.

    Maintains a steady pool of ready tasks by tracking the trend in ready-task
    count and, when the pool drifts, dispatching a high-priority task chosen by
    one of two static scores:

    - Inspiring **ability**: total downstream tasks dependent on this one
      (long-term utility).
    - Inspiring **efficiency**: how many children could complete within a time
      window (short-term utility).

    Owns all Inspirit-specific state (frontiers, the ready-count accumulator, the
    INC/DEC state machine) so it can compose with any Environment — the base
    Environment for batch schedulers like HEFT, or a FrontierEnvironment when a
    fill policy handles the remaining slots.

    Args:
        smoothing_rate: EMA smoothing factor for the k_inc rate estimate.
        delta_ready:    Cumulative ready-count change that triggers a dispatch
                        evaluation. Defaults to the number of workers.
        dec_step:       Steps-below-peak threshold for entering the DEC state.
                        Defaults to the number of workers.
        s_inc:          Ready-count delta threshold to trigger in the INC state.
                        Defaults to the number of workers.
        s_dec:          Ready-count delta per DEC-phase dispatch band.
                        Defaults to the number of workers.
        fill_policy:    Optional policy used to fill remaining available slots
                        after pinning a priority task. Use with a
                        FrontierEnvironment (e.g. FrontierFillPolicy). When None,
                        env.scheduler.schedule() fills the rest.
    """

    INC = "INC"
    DEC = "DEC"

    def __init__(
        self,
        smoothing_rate: float = 0.8,
        delta_ready: Optional[int] = None,
        dec_step: Optional[int] = None,
        s_inc: Optional[int] = None,
        s_dec: Optional[int] = None,
        fill_policy: Optional[OnlinePolicy] = None,
    ) -> None:
        if not 0 <= smoothing_rate <= 1:
            raise ValueError("smoothing_rate must be between 0 and 1")
        self._smoothing_rate = smoothing_rate
        self._delta_ready_override = delta_ready
        self._dec_step_override = dec_step
        self._s_inc_override = s_inc
        self._s_dec_override = s_dec
        self.fill_policy = fill_policy

        self._insertion_strategy = GreedyInsert(
            append_only=False,
            compare=GreedyInsertCompareFuncs.EFT,
            critical_path=False,
        )

        # Static scores — computed lazily on first update() call.
        self.efficiency_ranks: Optional[Dict[str, float]] = None
        self.ability_ranks: Optional[Dict[str, float]] = None
        self._time_window: Optional[float] = None

        # Per-run bookkeeping — reset by reset().
        self.efficiency_frontier: List[Tuple[float, str]] = []
        self.ability_frontier: List[Tuple[float, str]] = []
        self.peak: int = 0
        self.cur_state: Optional[str] = None
        self.dec_step: int = 0
        self.delta_ready: int = 0
        self.k_inc: float = 0.0
        self.cur_k: float = 0.0
        self.s_inc: int = 0
        self.s_dec: int = 0
        self.s_dec_count: int = 0
        self.c: int = 1
        self.last_dispatched: Optional[str] = None
        self.last_dispatch_type: Optional[str] = None
        self._last_k_time: float = 0.0
        self._last_k_nready: int = 0
        # Ready-count accumulator (replaces the former ReadyChangeObserver).
        self._last_ready_count: int = 0
        self._cumulative_delta: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear per-run bookkeeping. Static ranks are preserved across runs."""
        self.efficiency_frontier = []
        self.ability_frontier = []
        self.peak = 0
        self.cur_state = None
        self.dec_step = 0
        self.k_inc = 0.0
        self.cur_k = 0.0
        self.s_inc = 0
        self.s_dec = 0
        self.s_dec_count = 0
        self.c = 1
        self.last_dispatched = None
        self.last_dispatch_type = None
        self._last_k_time = 0.0
        self._last_k_nready = 0
        self._last_ready_count = 0
        self._cumulative_delta = 0
        if self.fill_policy is not None:
            self.fill_policy.reset()

    def _init_ranks(self, environment: "Environment") -> None:
        """Compute static efficiency/ability ranks and worker-count defaults."""
        workers = len(environment.network.nodes)
        time_window = workers * (
            np.mean([task.cost for task in environment.task_graph.tasks])
            / np.mean([node.speed for node in environment.network.nodes])
        )
        self._time_window = time_window
        self.efficiency_ranks = compute_inspiring_effeciency(
            environment.task_graph, environment.network, time_window=time_window
        )
        self.ability_ranks = compute_inspiring_ability(environment.task_graph)
        self.delta_ready = self._delta_ready_override if self._delta_ready_override is not None else workers
        self.dec_step = self._dec_step_override if self._dec_step_override is not None else workers
        self.s_inc = self._s_inc_override if self._s_inc_override is not None else workers
        self.s_dec = self._s_dec_override if self._s_dec_override is not None else workers
        self.c = max(1, workers // 2)

    # ------------------------------------------------------------------
    # Frontier management
    # ------------------------------------------------------------------

    def _rebuild_frontiers(self, environment: "Environment") -> None:
        """Rebuild efficiency and ability heaps from the current ready_tasks."""
        if self.efficiency_ranks is None or self.ability_ranks is None:
            raise RuntimeError("Inspirit ranks must be initialized before rebuilding frontiers.")
        ready_names = {t.name for t in environment.ready_tasks}
        self.efficiency_frontier = [
            (-self.efficiency_ranks[name], name) for name in ready_names
        ]
        self.ability_frontier = [
            (-self.ability_ranks[name], name) for name in ready_names
        ]
        heapq.heapify(self.efficiency_frontier)
        heapq.heapify(self.ability_frontier)

    def pop_highest_efficiency(self) -> Optional[str]:
        if self.efficiency_frontier:
            _, name = heapq.heappop(self.efficiency_frontier)
            return name
        return None

    def pop_highest_ability(self) -> Optional[str]:
        if self.ability_frontier:
            _, name = heapq.heappop(self.ability_frontier)
            return name
        return None

    # ------------------------------------------------------------------
    # Rate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_k(cur_nready: int, prev_nready: int, cur_time: float, prev_time: float) -> float:
        dt = cur_time - prev_time
        if dt <= 0:
            return 0.0
        return (cur_nready - prev_nready) / dt

    def _update_adaptive_k(self, cur_k: float, k_inc: float) -> float:
        alpha = self._smoothing_rate
        return alpha * k_inc + (1 - alpha) * cur_k

    # ------------------------------------------------------------------
    # Insertion helper
    # ------------------------------------------------------------------

    def _insert_task(self, task_name: str, schedule: Schedule, environment: "Environment") -> None:
        self._insertion_strategy.call(
            environment.network,
            environment.task_graph,
            schedule,
            task_name,
            min_start_time=environment.current_time,
        )

    def _accumulate_ready_change(self, environment: "Environment") -> bool:
        """Track the cumulative change in ready-count; return True when it crosses delta_ready.

        Replaces the former ReadyChangeObserver: accumulates signed changes in the
        ready-task count and fires (resetting the accumulator) once the magnitude
        reaches delta_ready.
        """
        current_ready = len(environment.ready_tasks)
        self._cumulative_delta += current_ready - self._last_ready_count
        self._last_ready_count = current_ready
        if abs(self._cumulative_delta) >= self.delta_ready:
            self._cumulative_delta = 0
            return True
        return False

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, environment: "Environment") -> Optional[Schedule]:
        env = environment

        if self.efficiency_ranks is None:
            self._init_ranks(env)

        # Clear per-step transient state so on_step callbacks see clean values.
        self.last_dispatched = None
        self.last_dispatch_type = None

        evaluate_dispatch = self._accumulate_ready_change(env)
        self._rebuild_frontiers(env)

        cur_nready = len(env.ready_tasks)
        task_name: Optional[str] = None
        dispatch_type: Optional[str] = None

        if evaluate_dispatch:
            if cur_nready > self.peak:
                self.peak = cur_nready
            if cur_nready >= self.peak - self.dec_step:
                self.cur_state = self.INC
            else:
                self.cur_state = self.DEC

            if self.cur_state == self.INC:
                if cur_nready - env.prev_nready > self.s_inc:
                    self.cur_k = self._calculate_k(
                        cur_nready, self._last_k_nready,
                        env.current_time, self._last_k_time,
                    )
                    self._last_k_time = env.current_time
                    self._last_k_nready = cur_nready
                    if self.cur_k < self.k_inc:
                        task_name = self.pop_highest_efficiency()
                        dispatch_type = "efficiency"
                    else:
                        task_name = self.pop_highest_ability()
                        dispatch_type = "ability"
                    self.k_inc = self._update_adaptive_k(self.cur_k, self.k_inc)
            elif self.cur_state == self.DEC:
                if cur_nready > self.peak - self.s_dec * self.s_dec_count:
                    task_name = self.pop_highest_ability()
                    dispatch_type = "ability"
                elif cur_nready <= self.peak - self.s_dec * (self.s_dec_count + 1) + self.c:
                    task_name = self.pop_highest_efficiency()
                    dispatch_type = "efficiency"
                    if cur_nready <= self.peak - self.s_dec * (self.s_dec_count + 1):
                        self.s_dec_count += 1

        self.last_dispatched = task_name
        self.last_dispatch_type = dispatch_type if task_name is not None else None

        if self.fill_policy is not None:
            # Frontier path: pin the priority task directly into env.schedule (removing
            # it from the frontier so the fill policy won't re-pick it), then always
            # let the fill policy fill the remaining available slots.
            if not isinstance(env, FrontierEnvironment):
                raise TypeError("InspiritPolicy with a fill_policy requires a FrontierEnvironment.")
            if task_name is not None:
                env.frontier_set.discard(task_name)
                env.frontier = [(p, n) for p, n in env.frontier if n != task_name]
                heapq.heapify(env.frontier)
                self._insert_task(task_name, env.schedule, env)
            return self.fill_policy.update(env)

        # Batch scheduler path (e.g. HEFT): if no dispatch, leave schedule unchanged.
        if task_name is None:
            return None

        partial = _build_partial_schedule(env)
        self._insert_task(task_name, partial, env)
        if env.scheduler is None:
            raise ValueError("InspiritPolicy requires environment.scheduler to be set.")
        return env.scheduler.schedule(
            env.network,
            env.task_graph,
            schedule=partial,
            min_start_time=env.current_time,
        )


class FrontierFillPolicy(OnlinePolicy):
    """Pops tasks from the frontier to fill all currently available nodes each step."""

    def __init__(self, insertion_strategy: Optional[GreedyInsert] = None):
        self._insertion_strategy = (
            insertion_strategy
            if insertion_strategy is not None
            else GreedyInsert(append_only=False, compare=GreedyInsertCompareFuncs.EST, critical_path=False)
        )

    def _insert_task(self, task_name: str, schedule: Schedule, env: "FrontierEnvironment") -> None:
        self._insertion_strategy.call(
            env.network,
            env.task_graph,
            schedule,
            task_name,
            min_start_time=env.current_time,
        )

    def update(self, environment: "Environment") -> Optional[Schedule]:
        if not isinstance(environment, FrontierEnvironment):
            raise ValueError("FrontierFillPolicy requires a FrontierEnvironment.")
        env = environment
        if not env.frontier:
            return None
        ready_node_count = len(env.available_nodes) if env.ready_node_only else len(env.frontier)
        for _ in range(ready_node_count):
            if not env.frontier:
                break
            _, task_name = heapq.heappop(env.frontier)
            env.frontier_set.discard(task_name)
            self._insert_task(task_name, env.schedule, env)
        return env.schedule
