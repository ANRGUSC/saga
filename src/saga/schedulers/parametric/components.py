from copy import deepcopy
import heapq
import numpy as np
from pydantic import BaseModel, Field
from enum import Enum

from saga import NetworkNode, Scheduler, ScheduledTask, TaskGraphNode
from saga.schedulers.parametric import IntialPriority, InsertTask, ParametricScheduler
from saga.schedulers.heft import heft_rank_sort
from saga.schedulers.cpop import cpop_ranks
from saga import Network, TaskGraph, Schedule

import networkx as nx
from typing import Dict, Iterable, List, Hashable, Literal


class UpwardRanking(IntialPriority):
    name: Literal["UpwardRanking"] = "UpwardRanking"

    def call(self, network: Network, task_graph: TaskGraph) -> List[str]:
        return heft_rank_sort(network, task_graph)


class CPoPRanking(IntialPriority):
    def call(self, network: Network, task_graph: TaskGraph) -> List[str]:
        ranks = cpop_ranks(network, task_graph)
        start_task = max(
            [task.name for task in task_graph.tasks if task_graph.in_degree(task) == 0],
            key=lambda t: ranks[t],
        )
        pq = [(-ranks[start_task], start_task)]
        heapq.heapify(pq)
        queue = []
        while pq:
            _, task_name = heapq.heappop(pq)
            queue.append(task_name)
            ready_tasks = [
                dependency.target
                for dependency in task_graph.out_edges(task_name)
                if all(
                    succ_dependency.source in queue
                    for succ_dependency in task_graph.in_edges(dependency.target)
                )
            ]
            for ready_task in ready_tasks:
                heapq.heappush(pq, (-ranks[ready_task], ready_task))

        return queue


class ArbitraryTopological(IntialPriority):
    def call(self, network: Network, task_graph: TaskGraph) -> List[str]:
        return [task.name for task in task_graph.topological_sort()]


class GreedyInsertCompareFuncs(Enum):
    EFT = "EFT"
    EST = "EST"
    Quickest = "Quickest"

    def compare(self, new: ScheduledTask, cur: ScheduledTask) -> float:
        if self == GreedyInsertCompareFuncs.EFT:
            return new.end - cur.end
        elif self == GreedyInsertCompareFuncs.EST:
            return new.start - cur.start
        elif self == GreedyInsertCompareFuncs.Quickest:
            return (new.end - new.start) - (cur.end - cur.start)
        else:
            raise ValueError(f"Unknown comparison function: {self.value}")


# Insert Task functions
class GreedyInsert(InsertTask):
    append_only: bool = Field(
        False, description="Whether to only append the task to the schedule."
    )
    compare: GreedyInsertCompareFuncs = Field(
        GreedyInsertCompareFuncs.EFT,
        description="The comparison function to use for greedy insertion.",
    )
    critical_path: bool = Field(
        False, description="Whether to only schedule tasks on the critical path."
    )

    def _compare(self, new: ScheduledTask, cur: ScheduledTask) -> float:
        return self.compare.compare(new, cur)

    def call(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Schedule,
        task: str | TaskGraphNode,
        min_start_time: float = 0.0,
        nodes: Iterable[str] | Iterable[NetworkNode] | None = None,
        dry_run: bool = False,
    ) -> ScheduledTask:
        task = task_graph.get_task(task)
        if nodes is None:
            nodes = network.nodes
        considered_nodes = [
            node if isinstance(node, NetworkNode) else network.get_node(node)
            for node in nodes
        ]

        best_task = None
        if self.critical_path:
            ranks = cpop_ranks(network, task_graph)
            critical_rank = max(ranks.values())
            fastest_node = max(network.nodes, key=lambda node: node.speed)
            if np.isclose(ranks[task.name], critical_rank):
                considered_nodes = [fastest_node]

        for _node in considered_nodes:
            exec_time = task.cost / _node.speed
            start_time = schedule.get_earliest_start_time(
                task=task,
                node=_node,
                append_only=self.append_only,
            )
            start_time = max(start_time, min_start_time)

            new_task = ScheduledTask(
                node=_node.name,
                name=task.name,
                start=start_time,
                end=start_time + exec_time,
            )
            if best_task is None or self._compare(new_task, best_task) < 0:
                best_task = new_task

        if best_task is None:
            raise RuntimeError("No valid task insertion found.")
        if not dry_run:
            schedule.add_task(best_task)
        return best_task


class ParametricKDepthScheduler(Scheduler, BaseModel):
    scheduler: ParametricScheduler = Field(
        ..., description="The base parametric scheduler."
    )
    k_depth: int = Field(
        ..., description="The depth of the subgraph to consider for scheduling."
    )

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Schedule | None = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Schedule the tasks on the network.

        Args:
            network (Network): The network graph.
            task_graph (TaskGraph): The task graph.
            schedule (Optional[Schedule]): The current schedule.

        Returns:
            Schedule: The resulting schedule.
        """
        queue = self.scheduler.initial_priority.call(network, task_graph)
        schedule = (
            Schedule(task_graph, network) if schedule is None else deepcopy(schedule)
        )
        scheduled_tasks: Dict[Hashable, ScheduledTask] = {}
        while queue:
            task_name = queue.pop(0)
            k_depth_successors = nx.single_source_shortest_path_length(
                G=task_graph.graph, source=task_name, cutoff=self.k_depth
            )
            sub_task_graph = task_graph.graph.subgraph(
                set(scheduled_tasks.keys())
                | set(k_depth_successors.keys())
                | {task_name}
            )

            best_node, best_makespan = None, float("inf")
            for node in network.nodes:
                sub_schedule = deepcopy(schedule)
                _network = deepcopy(network)
                _task_graph = deepcopy(task_graph)
                _sub_task_graph = TaskGraph.from_nx(sub_task_graph.copy().to_directed())
                self.scheduler.insert_task.call(
                    network=_network,
                    task_graph=_task_graph,
                    schedule=sub_schedule,
                    task=task_name,
                    min_start_time=min_start_time,
                    nodes=[node],
                )
                sub_schedule = self.scheduler.schedule(
                    network=_network,
                    task_graph=_sub_task_graph,
                    schedule=sub_schedule,
                    min_start_time=min_start_time,
                )
                if schedule.makespan < best_makespan:
                    best_node = node
                    best_makespan = schedule.makespan

            if best_node is None:
                raise RuntimeError("No valid node found for task scheduling.")
            task = self.scheduler.insert_task.call(
                network=network,
                task_graph=task_graph,
                schedule=schedule,
                task=task_name,
                min_start_time=min_start_time,
                nodes=[best_node],
            )
            scheduled_tasks[task.name] = task

        return schedule

    @property
    def name(self) -> str:
        return f"{self.scheduler.name}_KDepth{self.k_depth}"


class ParametricSufferageScheduler(ParametricScheduler):
    insert_task: GreedyInsert = Field(..., description="The task insertion strategy.")
    scheduler: ParametricScheduler = Field(
        ..., description="The base parametric scheduler."
    )
    top_n: int = Field(
        2, description="The number of top tasks to consider for sufferage calculation."
    )

    def __init__(self, scheduler: ParametricScheduler, top_n: int = 2) -> None:
        super().__init__(
            initial_priority=scheduler.initial_priority,
            insert_task=scheduler.insert_task,
            scheduler=scheduler,
            top_n=top_n,
        )

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Schedule | None = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Schedule the tasks on the network.

        Args:
            network (Network): The network graph.
            task_graph (TaskGraph): The task graph.
            schedule (Optional[Schedule]): The current schedule.
            min_start_time (float): The current moment in time.

        Returns:
            Schedule: The resulting schedule.
        """
        queue = self.initial_priority.call(network, task_graph)
        if schedule is None:
            schedule = Schedule(task_graph, network)
        while queue:
            for task_name in queue:
                if schedule.is_scheduled(task_name):
                    queue.remove(task_name)

            ready_tasks = [
                task
                for task in queue
                if all(
                    schedule.is_scheduled(dep.source)
                    for dep in task_graph.in_edges(task)
                )
            ]
            max_sufferage_task, max_sufferage = None, -np.inf
            for task_name in ready_tasks[: self.top_n]:
                best_task = self.insert_task.call(
                    network,
                    task_graph,
                    schedule,
                    task_name,
                    min_start_time,
                    dry_run=True,
                )
                second_best_task = self.insert_task.call(
                    network=network,
                    task_graph=task_graph,
                    schedule=schedule,
                    task=task_name,
                    min_start_time=min_start_time,
                    nodes=[
                        node for node in network.nodes if node.name != best_task.node
                    ],
                    dry_run=True,
                )
                sufferage = self.insert_task._compare(second_best_task, best_task)
                if sufferage > max_sufferage:
                    max_sufferage_task, max_sufferage = best_task, sufferage

            if max_sufferage_task is None:
                raise RuntimeError("No ready tasks to schedule.")
            new_task = self.insert_task.call(
                network=network,
                task_graph=task_graph,
                schedule=schedule,
                task=max_sufferage_task.name,
                min_start_time=min_start_time,
                nodes=[max_sufferage_task.node],
            )
            queue.remove(new_task.name)

        return schedule

    @property
    def name(self) -> str:
        return f"{self.scheduler.name}_Sufferage"


insert_funcs: Dict[str, GreedyInsert] = {}
for compare_func in GreedyInsertCompareFuncs:
    insert_funcs[f"{compare_func.value}_Append"] = GreedyInsert(
        append_only=True, compare=compare_func, critical_path=False
    )
    insert_funcs[f"{compare_func.value}_Insert"] = GreedyInsert(
        append_only=False, compare=compare_func, critical_path=False
    )
    insert_funcs[f"{compare_func.value}_Append_CP"] = GreedyInsert(
        append_only=True, compare=compare_func, critical_path=True
    )
    insert_funcs[f"{compare_func.value}_Insert_CP"] = GreedyInsert(
        append_only=False, compare=compare_func, critical_path=True
    )

initial_priority_funcs = {
    "UpwardRanking": UpwardRanking(),
    "CPoPRanking": CPoPRanking(),
    "ArbitraryTopological": ArbitraryTopological(),
}

schedulers: Dict[str, ParametricScheduler] = {}
for name, insert_func in insert_funcs.items():
    for intial_priority_name, initial_priority_func in initial_priority_funcs.items():
        reg_scheduler = ParametricScheduler(
            initial_priority=initial_priority_func, insert_task=insert_func
        )
        schedulers[reg_scheduler.name] = reg_scheduler

        sufferage_scheduler = ParametricSufferageScheduler(
            scheduler=reg_scheduler, top_n=2
        )
        schedulers[sufferage_scheduler.name] = sufferage_scheduler
