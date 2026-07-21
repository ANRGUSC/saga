"""FrontierFillPolicy: pop frontier tasks to fill available nodes each step."""

from __future__ import annotations

import heapq
from typing import Optional, TYPE_CHECKING

from saga import Schedule
from saga.schedulers.online.policy import OnlinePolicy
from saga.schedulers.online.environment import FrontierEnvironment
from saga.schedulers.parametric.components import GreedyInsert, GreedyInsertCompareFuncs

if TYPE_CHECKING:
    from saga.schedulers.online.environment import Environment


class FrontierFillPolicy(OnlinePolicy):
    """Pops tasks from the frontier to fill all currently available nodes each step."""

    def __init__(self, insertion_strategy: Optional[GreedyInsert] = None):
        self._insertion_strategy = (
            insertion_strategy
            if insertion_strategy is not None
            else GreedyInsert(
                append_only=False,
                compare=GreedyInsertCompareFuncs.EST,
                critical_path=False,
            )
        )

    def _insert_task(
        self, task_name: str, schedule: Schedule, env: "FrontierEnvironment"
    ) -> None:
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
        ready_node_count = (
            len(env.available_nodes) if env.ready_node_only else len(env.frontier)
        )
        for _ in range(ready_node_count):
            if not env.frontier:
                break
            _, task_name = heapq.heappop(env.frontier)
            env.frontier_set.discard(task_name)
            self._insert_task(task_name, env.schedule, env)
        return env.schedule
