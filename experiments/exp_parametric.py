from copy import deepcopy
import heapq
import logging
import pathlib
import shutil

import numpy as np
from saga.scheduler import Scheduler, Task
from saga.schedulers.parametric import IntialPriority, ScheduleType, UpdatePriority, InsertTask, ParametricScheduler, ParametricKDepthScheduler
from saga.schedulers.heft import heft_rank_sort, get_insert_loc
from saga.schedulers.cpop import cpop_ranks

import networkx as nx
from typing import Callable, List, Hashable

from saga.utils.testing import test_schedulers

# Initial Priority functions
class UpwardRanking(IntialPriority):
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> List[Hashable]:
        return heft_rank_sort(network, task_graph)

class CPoPRanking(IntialPriority):
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> List[Hashable]:
        ranks = cpop_ranks(network, task_graph)
        start_task = max(
            [task for task in task_graph if task_graph.in_degree(task) == 0],
            key=ranks.get
        )
        pq = [(-ranks[start_task], start_task)]
        heapq.heapify(pq)
        queue = []
        while pq:
            _, task_name = heapq.heappop(pq)
            queue.append(task_name)
            ready_tasks = [
                succ for succ in task_graph.successors(task_name)
                if all(pred in queue for pred in task_graph.predecessors(succ))
            ]
            for ready_task in ready_tasks:
                heapq.heappush(pq, (-ranks[ready_task], ready_task))
        
        return queue

class ArbitraryTopological(IntialPriority):
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> List[Hashable]:
        return list(nx.topological_sort(task_graph))

# Insert Task functions
class GreedyInsert(InsertTask):
    def __init__(self,
                 append_only: bool = False,
                 compare: Callable[[Task, Task], float] = lambda new, cur: new.end - cur.end):
        """Initialize the GreedyInsert class.
        
        Args:
            append_only (bool, optional): Whether to only append the task to the schedule. Defaults to False.
            compare (Callable[[Task, Task], float], optional): The comparison function. Defaults to lambda new, cur: new.end - cur.end.
                The comparison function should return a negative number if the first task is better than the second task, a positive number
                if the second task is better than the first task, and 0 if the tasks are equally good.
        """
        self.append_only = append_only
        self.compare = compare

    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: ScheduleType,
                 task: Hashable) -> Task:
        best_insert_loc, best_task = None, None
        # TODO: This is a bit inefficient adds O(n) per iteration,
        #       so it doesn't affect asymptotic complexity, but it's still bad
        task_map = {
            task.name: task
            for node, tasks in schedule.items()
            for task in tasks
        }
        # print(task_map)
        for node in network.nodes:
            # print(f"Considering scheduling {task} on {node}")
            exec_time = task_graph.nodes[task]['weight'] / network.nodes[node]['weight']

            min_start_time = 0
            for parent in task_graph.predecessors(task):
                parent_node = task_map[parent].node
                data_size = task_graph.edges[parent, task]["weight"]
                comm_strength = network.edges[parent_node, node]["weight"]
                comm_time = data_size / comm_strength
                min_start_time = max(min_start_time, task_map[parent].end + comm_time)

            if self.append_only:
                start_time = max(
                    min_start_time,
                    0.0 if not schedule[node] else schedule[node][-1].end
                )
                insert_loc = len(schedule[node])
            else:
                insert_loc, start_time  = get_insert_loc(schedule[node], min_start_time, exec_time)

            new_task = Task(node, task, start_time, start_time + exec_time)
            if best_task is None or self.compare(new_task, best_task) < 0:
                best_insert_loc, best_task = insert_loc, new_task

        schedule[best_task.node].insert(best_insert_loc, best_task)
        return best_task

# Update Priority functions
class NoUpdate(UpdatePriority):
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: ScheduleType,
                 queue: List[Hashable]) -> List[Hashable]:
        return queue

class SufferageUpdatePriority(UpdatePriority):
    def __init__(self, insert_task: GreedyInsert):
        self.insert_task = insert_task

    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: ScheduleType,
                 queue: List[Hashable]) -> List[Hashable]:
        sufferages = {}
        scheduled_tasks = {
            task.name: task
            for node, tasks in schedule.items()
            for task in tasks
        }
        for task in queue:
            if not all(pred in scheduled_tasks for pred in task_graph.predecessors(task)):
                sufferages[task] = -np.inf
                continue

            best_task = self.insert_task(network, task_graph, deepcopy(schedule), task)
            _network = network.copy()
            _network.nodes[best_task.node]['weight'] = 1e-9
            second_best_task = self.insert_task(
                _network,
                task_graph,
                deepcopy(schedule),
                task
            )
            sufferages[task] = self.insert_task.compare(second_best_task, best_task)
            if sufferages[task] < 0:
                raise ValueError(f"Task {task} has negative sufferage {sufferages[task]}")
        return sorted(queue, key=sufferages.get, reverse=True)


compare_funcs = {
    "EFT": lambda new, cur: new.end - cur.end,
    "EST": lambda new, cur: new.start - cur.start,
    "Quickest": lambda new, cur: new.end - cur.end,
}
insert_funcs = {}
for name, compare_func in compare_funcs.items():
    insert_funcs[f"{name}_Append"] = GreedyInsert(append_only=True, compare=compare_func)
    insert_funcs[f"{name}_Insert"] = GreedyInsert(append_only=False, compare=compare_func)

initial_priority_funcs = {
    "UpwardRanking": UpwardRanking(),
    "CPoPRanking": CPoPRanking(),
    "ArbitraryTopological": ArbitraryTopological()
}

schedulers = {}
for name, insert_func in insert_funcs.items():
    for intial_priority_name, initial_priority_func in initial_priority_funcs.items():
        schedulers[f"{name}_{intial_priority_name}"] = ParametricScheduler(
            initial_priority=initial_priority_func,
            insert_task=insert_func,
            update_priority=NoUpdate()
        )
        schedulers[f"{name}_Sufferage_{intial_priority_name}"] = ParametricScheduler(
            initial_priority=initial_priority_func,
            insert_task=insert_func,
            update_priority=SufferageUpdatePriority(insert_func)
        )

        for k in range(1, 4):
            schedulers[f"{name}_{intial_priority_name}_K{k}"] = ParametricKDepthScheduler(
                scheduler=schedulers[f"{name}_{intial_priority_name}"],
                k_depth=k
            )
            schedulers[f"{name}_Sufferage_{intial_priority_name}_K{k}"] = ParametricKDepthScheduler(
                scheduler=schedulers[f"{name}_Sufferage_{intial_priority_name}"],
                k_depth=k
            )


def main():
    logging.basicConfig(level=logging.DEBUG)

    thisdir = pathlib.Path(__file__).parent.resolve()
    savedir = thisdir / "results" / "parametric" / "tests"
    if savedir.exists():
        shutil.rmtree(savedir)
    savedir.mkdir(exist_ok=True, parents=True)

    # scheduler_names_to_test = [
    #     "EST_Insert_CPoPRanking",
    # ]
    # for scheduler_name_to_test in scheduler_names_to_test:
    #     test_schedulers(
    #         {scheduler_name_to_test: schedulers[scheduler_name_to_test]},
    #         savedir=savedir,
    #         stop_on_error=True
    #     )

    test_schedulers(schedulers, savedir=savedir, stop_on_error=True)
    

if __name__ == "__main__":
    main()