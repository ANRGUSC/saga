"""Frontier-based online environment (FIFO, LIFO, FrontierHEFT, ...)."""

from __future__ import annotations

import heapq
from typing import Any, Callable, List, Optional, Set, Tuple, TYPE_CHECKING

from saga import Network, Schedule, Scheduler, TaskGraph, TaskGraphNode
from saga.schedulers.online.environment import (
    Environment,
    StepFunction,
    next_event,
)
from saga.schedulers.parametric.components import GreedyInsert, GreedyInsertCompareFuncs

if TYPE_CHECKING:
    from saga.schedulers.online.policy import OnlinePolicy


class FrontierEnvironment(Environment):
    """Environment for frontier-based online scheduling. Algorithms include FIFO, LIFO, FrontierHEFT, etc."""

    def __init__(
        self,
        network: Network,
        task_graph: TaskGraph,
        scheduler: Optional[Scheduler] = None,
        policy: Optional["OnlinePolicy"] = None,
        step: Optional[StepFunction] = None,
        on_step: Optional[Callable[["Environment"], None]] = None,
    ) -> None:
        super().__init__(
            network=network,
            task_graph=task_graph,
            scheduler=scheduler,
            policy=policy,
            step=step if step is not None else next_event,
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
        self.priority_condition: Callable[[TaskGraphNode], Any] = lambda _: (
            self.current_time
        )

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
        self.unready_tasks = {task.name for task in self.task_graph.tasks}
        self.available_nodes = set()
        self.occupied_nodes = set()
        self._step = 0
        self.history = []
        if self.policy is not None:
            self.policy.reset()
        self.frontier = []
        self.frontier_set = set()
        if schedule is not None:
            self.schedule = schedule
        else:
            self.schedule = Schedule(
                task_graph=self.task_graph,
                network=self.network,
                node_constraints=self.node_constraints,
            )
            for task in self.task_graph.tasks:
                if self.task_graph.in_degree(task) == 0:
                    self._bootstrap_insert.call(
                        self.network,
                        self.task_graph,
                        self.schedule,
                        task.name,
                        min_start_time=min_start_time,
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
                    heapq.heappush(self.frontier, (self.priority_condition(task), name))
            elif self.ready_condition == "p_committed":
                if predecessors.issubset(finished_or_running):
                    self.frontier_set.add(name)
                    heapq.heappush(self.frontier, (self.priority_condition(task), name))
            elif self.ready_condition == "p_scheduled":
                if predecessors.issubset(committed_names):
                    self.frontier_set.add(name)
                    heapq.heappush(self.frontier, (self.priority_condition(task), name))

        task_by_name = {task.name: task for task in self.task_graph.tasks}
        self.ready_tasks = {task_by_name[name] for name in self.frontier_set}
        self.unready_tasks = {
            task.name
            for task in self.task_graph.tasks
            if task.name not in committed_names and task.name not in self.frontier_set
        }
