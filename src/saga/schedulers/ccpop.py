"""Conditional CPOP (CCPOP) scheduler.

Pre-weights task costs and edge sizes by compound probability, then runs
standard CPOP ranking (upward + downward rank).  Tasks on high-probability
paths get higher combined rank and therefore better node placements.

When the task graph has no conditional edges, all probabilities are 1.0 and
CCPOP degrades to standard CPOP.
"""

import heapq
from typing import Dict, Optional

import numpy as np

from saga import Network, Schedule, ScheduledTask, Scheduler, TaskGraph
from saga.schedulers.cheft import build_weighted_graph
from saga.schedulers.cpop import upward_rank, downward_rank
from saga.utils.duplication import should_duplicate


def ccpop_ranks(network: Network, task_graph: TaskGraph) -> Dict[str, float]:
    """CPOP ranks (upward + downward) computed on the probability-weighted graph."""
    weighted = build_weighted_graph(task_graph)
    up = upward_rank(network, weighted)
    down = downward_rank(network, weighted)
    return {t.name: up[t.name] + down[t.name] for t in task_graph.tasks}


class CCpopScheduler(Scheduler):
    """CPOP with probability-weighted ranking for conditional task graphs."""

    duplication_factor: int = 1

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        comp_schedule = Schedule(task_graph, network)
        task_map: Dict[str, ScheduledTask] = {}
        if schedule is not None:
            comp_schedule = schedule.model_copy()
            task_map = {
                task.name: task for _, tasks in schedule.items() for task in tasks
            }

        ranks = ccpop_ranks(network, task_graph)
        entry_tasks = [
            task.name
            for task in task_graph.tasks
            if task_graph.in_degree(task.name) == 0
        ]
        cp_rank = ranks[max(entry_tasks, key=lambda t: ranks[t])]

        cp_node = min(
            network.nodes,
            key=lambda node: sum(
                task.cost / node.speed
                for task in task_graph.tasks
                if np.isclose(ranks[task.name], cp_rank)
            ),
        )

        pq = [
            (-ranks[task.name], task)
            for task in task_graph.tasks
            if task.name not in task_map
            and all(
                dep.source in task_map
                for dep in task_graph.in_edges(task.name)
            )
        ]
        heapq.heapify(pq)

        while pq:
            task_rank, task = heapq.heappop(pq)

            is_critical = np.isclose(-task_rank, cp_rank)
            nodes = frozenset([cp_node]) if is_critical else network.nodes

            duplicate_factor = 1
            if not is_critical and should_duplicate(task.name, task_graph, network):
                duplicate_factor = self.duplication_factor

            for dup_idx in range(duplicate_factor):
                best_node = None
                min_finish_time = np.inf
                for node in nodes:
                    start_time = comp_schedule.get_earliest_start_time(
                        task=task, node=node, append_only=False
                    )
                    end_time = start_time + (task.cost / node.speed)
                    if end_time < min_finish_time:
                        min_finish_time = end_time
                        best_node = node

                new_task = ScheduledTask(
                    node=best_node.name,
                    name=task.name,
                    start=min_finish_time - (task.cost / best_node.speed),
                    end=min_finish_time,
                )
                comp_schedule.add_task(new_task)
                task_map[task.name] = new_task

            ready_tasks = [
                task_graph.get_task(dep.target)
                for dep in task_graph.out_edges(task.name)
                if all(
                    child_dep.source in task_map
                    for child_dep in task_graph.in_edges(dep.target)
                )
            ]
            for ready_task in ready_tasks:
                heapq.heappush(pq, (-ranks[ready_task.name], ready_task))

        return comp_schedule
