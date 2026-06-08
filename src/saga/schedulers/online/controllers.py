import logging
from copy import deepcopy
from typing import List, Optional, TYPE_CHECKING
import heapq

from saga import Network, Schedule, ScheduledTask, Scheduler, TaskGraph
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.online.components import Controller, Trigger
from saga.schedulers.online.environments import InspiritEnvironment, StochasticEnvironment, FrontierEnvironment
from saga.schedulers.parametric.components import GreedyInsert, GreedyInsertCompareFuncs
from saga.schedulers.stochastic import EstimateStochasticScheduler
from saga.stochastic import (
    StochasticNetwork,
    StochasticTaskGraph,
    StochasticSchedule,
    StochasticScheduledTask,
    StochasticScheduler,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from saga.schedulers.online.environment import Environment


def _build_partial_schedule(environment: "Environment") -> Schedule:
    """Return a Schedule containing only the committed tasks.

    Committed tasks are those that are finished or currently running — they
    have fixed positions in time that the rescheduler must work around.
    Running tasks keep their original start/end times (deterministic case).
    """
    partial = Schedule(environment.task_graph, environment.network)
    if isinstance(environment, StochasticEnvironment):
        environment.schedule_actual = environment.estimate_schedule.determinize(environment.actual_network, environment.actual_task_graph)
        for task in environment.finished_tasks:
            partial.add_task(task)

        est_running_tasks:List[ScheduledTask] = deepcopy(environment.running_tasks)
       
        for task in est_running_tasks:
            est_task_size = environment.task_graph.get_task(task.name).cost
            est_network_speed = environment.network.get_node(task.node).speed
            task.end = task.start + (est_task_size/est_network_speed)
            partial.add_task(task)
        return partial
    else: 
        for task in environment.finished_tasks:
            partial.add_task(task)
        for task in environment.running_tasks:
            partial.add_task(deepcopy(task))
        return partial


class RescheduleController(Controller):
    """Reschedules all remaining tasks around committed (finished + running) tasks.

    Builds a partial schedule from the committed tasks, then calls the
    scheduler to fill in the rest, starting no earlier than current_time.

    By default uses env.scheduler (the same strategy that produced the initial
    schedule).  Pass an alternate ParametricScheduler to use a different
    strategy at reschedule time.

    Args:
        scheduler: Optional alternate scheduler.  If None, env.scheduler is used.
    """

    def control(self, environment: "Environment", trigger: Trigger) -> Schedule:
        if not isinstance(environment.scheduler, ParametricScheduler):
            logger.warning(
                "RescheduleController: env.scheduler is not a ParametricScheduler "
                "(%s). Rescheduling requires schedule and min_start_time support; "
                "this may raise at runtime.",
                type(environment.scheduler).__name__,
            )
        partial = _build_partial_schedule(environment)
        if isinstance(environment, StochasticEnvironment):
            new_estimate = environment.stochastic_scheduler.schedule(
                environment._stochastic_network,
                environment._stochatic_task_graph,
                schedule=partial,
                min_start_time=environment.current_time,
            )[0]
            environment.estimate_schedule = new_estimate
            new_schedule = new_estimate.determinize(
                environment.actual_network, environment.actual_task_graph
            )
            environment.schedule_actual = new_schedule
        else:
            new_schedule = environment.scheduler.schedule(
                environment.network,
                environment.task_graph,
                schedule=partial,
                min_start_time=environment.current_time,
            )
        return new_schedule
class InspiritController(Controller):
    """Controller based on the Inspirit paper's approach of monitoring ready tasks and pop specific tasks when triggerd,
    with the goal of maximizing throughput. """
    def __init__(self, smoothing_rate: float = 0.8, scheduler: Optional[Scheduler] = None) -> None:
        if not 0 <= smoothing_rate <= 1:
            raise ValueError("smoothing_rate must be between 0 and 1")
        self._alt_scheduler = scheduler
        self._smoothing_rate = smoothing_rate
        self._insertion_strategy = GreedyInsert(append_only=False,
                                                compare=GreedyInsertCompareFuncs.EFT,
                                                critical_path=False)

    @staticmethod
    def _calculate_k(cur_nready: int, prev_nready: int, cur_time: float, prev_time: float) -> float:
        dt = cur_time - prev_time
        if dt <= 0:
            return 0.0
        return (cur_nready - prev_nready) / dt

    def _update_adaptive_k(self, cur_k: float, k_inc: float) -> float:
        alpha = self._smoothing_rate
        return alpha * k_inc + (1 - alpha) * cur_k
    
    def _insert_task(self, task_name: str, schedule: Schedule, env: InspiritEnvironment) -> None:
        """Insert task_name into schedule at its earliest feasible slot."""
        self._insertion_strategy.call(
            env.network,
            env.task_graph,
            schedule,
            task_name,
            min_start_time=env.current_time,
        )
    

    def control(self, environment: "Environment", trigger: Trigger) -> Schedule:
        """Select a priority task (if threshold is met), lock it into the partial
        schedule, then reschedule all remaining tasks around it.
        """
        if not isinstance(environment, InspiritEnvironment):
            raise ValueError("InspiritController requires an InspiritEnvironment.")
        env: InspiritEnvironment = environment
        cur_nready = len(env.ready_tasks)
        task_name: Optional[str] = None

        if cur_nready > env.peak:
            env.peak = cur_nready
        if cur_nready >= env.peak - env.dec_step:
            env.cur_state = InspiritEnvironment.INC
        else:
            env.cur_state = InspiritEnvironment.DEC

        dispatch_type: Optional[str] = None
        if env.cur_state == InspiritEnvironment.INC:
            if cur_nready - env.prev_nready > env.s_inc:
                # Measure rate over the inter-dispatch interval (since last k update),
                # not the single step interval, to avoid blowup when task completions
                # are closely spaced in time.
                env.cur_k = self._calculate_k(
                    cur_nready, env._last_k_nready,
                    env.current_time, env._last_k_time,
                )
                env._last_k_time = env.current_time
                env._last_k_nready = cur_nready
                if env.cur_k < env.k_inc:
                    task_name = env.pop_highest_efficiency()
                    dispatch_type = "efficiency"
                else:
                    task_name = env.pop_highest_ability()
                    dispatch_type = "ability"
                env.k_inc = self._update_adaptive_k(env.cur_k, env.k_inc)
        elif env.cur_state == InspiritEnvironment.DEC:
            if cur_nready > env.peak - env.s_dec * env.s_dec_count:
                task_name = env.pop_highest_ability()
                dispatch_type = "ability"
            elif cur_nready <= env.peak - env.s_dec * (env.s_dec_count + 1) + env.c:
                task_name = env.pop_highest_efficiency()
                dispatch_type = "efficiency"
                if cur_nready <= env.peak - env.s_dec * (env.s_dec_count + 1):
                    env.s_dec_count += 1

        env.last_dispatched = task_name
        env.last_dispatch_type = dispatch_type if task_name is not None else None
        if task_name is not None:
            partial = _build_partial_schedule(env)
            self._insert_task(task_name, partial, env)

            scheduler = self._alt_scheduler or env.scheduler
            return scheduler.schedule(
                env.network,
                env.task_graph,
                schedule=partial,
                min_start_time=env.current_time,
            )
        return env.schedule  # No change

#need to confirm that multiple tasks can be popped and scheduled at once, if multiple available nodes.
class FrontierPopController(Controller):
    """Controller that pops a task from the frontier whenever triggered, without checking a threshold."""
    def __init__(self, insertion_strategy: Optional[GreedyInsert] = None):
        super().__init__()
        self._insertion_strategy = (
            insertion_strategy
            if insertion_strategy is not None
            else GreedyInsert(append_only=False, compare=GreedyInsertCompareFuncs.EST, critical_path=False)
        )

    def _insert_task(self, task_name: str, schedule: Schedule, env: "FrontierEnvironment") -> None:
        """Insert task_name into schedule at its earliest feasible slot."""
        self._insertion_strategy.call(
            env.network,
            env.task_graph,
            schedule,
            task_name,
            min_start_time=env.current_time,
        )

    def control(self, environment: "Environment", trigger: Trigger) -> Schedule:
        if not isinstance(environment, FrontierEnvironment):
            raise ValueError("FrontierPopController requires a FrontierEnvironment.")
        env = environment
        if not env.frontier:
            return env.schedule
        ready_node_count = len(environment.available_nodes)
        for _ in range(ready_node_count):
            if not env.frontier:
                break
            _, task_name = heapq.heappop(env.frontier)
            env.frontier_set.discard(task_name)
            self._insert_task(task_name, env.schedule, env)
        return env.schedule