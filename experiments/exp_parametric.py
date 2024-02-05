import argparse
from copy import deepcopy
from functools import lru_cache
import heapq
import logging
import pathlib
from pprint import pformat
import shutil
import time

import numpy as np
import pandas as pd
from experiments.exp_benchmarking import TrimmedDataset
from experiments.prepare_datasets import load_dataset
from saga.data import Dataset
from saga.scheduler import Scheduler, Task
from saga.schedulers.parametric import IntialPriority, ScheduleType, UpdatePriority, InsertTask, ParametricScheduler, ParametricKDepthScheduler
from saga.schedulers.heft import heft_rank_sort, get_insert_loc
from saga.schedulers.cpop import cpop_ranks

import networkx as nx
from typing import Callable, Dict, List, Hashable
from saga.utils.random_graphs import add_random_weights, get_branching_dag, get_chain_dag, get_diamond_dag, get_fork_dag, get_network

from saga.utils.testing import test_schedulers
from saga.utils.tools import standardize_task_graph

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

class CriticalPathGreedyInsert(GreedyInsert):
    def __init__(self, append_only: bool = False, compare: Callable[[Task, Task], float] = lambda new, cur: new.end - cur.end):
        super().__init__(
            append_only=append_only,
            compare=compare
        )

    @lru_cache(maxsize=None)
    def cpop_ranks(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
        return cpop_ranks(network, task_graph)

    def __call__(self, network: nx.Graph, task_graph: nx.DiGraph, schedule: ScheduleType, task: Hashable) -> Task:
        # Same as GreedyInsert, but commits to scheduling nodes on the critical path to the fastest node.
        ranks = self.cpop_ranks(network, task_graph)
        critical_rank = max(ranks.values())
        critical_tasks = {
            task for task, rank in ranks.items()
            if np.isclose(rank, critical_rank)
        }
        scheduled_tasks = {
            task.name: task
            for node, tasks in schedule.items()
            for task in tasks
        }

        if task in critical_tasks:
            # Assign to the node with the fastest critical path execution time
            best_node = max(
                network.nodes,
                key=lambda node: network.nodes[node]['weight']
            )
            exec_time = task_graph.nodes[task]['weight'] / network.nodes[best_node]['weight']
            min_start_time = 0
            for parent in task_graph.predecessors(task):
                parent_node = scheduled_tasks[parent].node
                data_size = task_graph.edges[parent, task]["weight"]
                comm_strength = network.edges[parent_node, best_node]["weight"]
                comm_time = data_size / comm_strength
                min_start_time = max(min_start_time, scheduled_tasks[parent].end + comm_time)
            
            if self.append_only:
                start_time = max(
                    min_start_time,
                    0.0 if not schedule[best_node] else schedule[best_node][-1].end
                )
                insert_loc = len(schedule[best_node])
            else:
                insert_loc, start_time  = get_insert_loc(schedule[best_node], min_start_time, exec_time)

            new_task = Task(best_node, task, start_time, start_time + exec_time)
            schedule[best_node].insert(insert_loc, new_task)
            return new_task
        else:
            return super().__call__(network, task_graph, schedule, task)

# Update Priority functions
class NoUpdate(UpdatePriority):
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: ScheduleType,
                 queue: List[Hashable]) -> List[Hashable]:
        return queue

class SufferageUpdatePriority(UpdatePriority):
    def __init__(self, insert_task: GreedyInsert, top_n: int = None):
        self.insert_task = insert_task
        self.top_n = top_n

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
        top_n = self.top_n if self.top_n is not None else len(queue)
        for task in queue[:top_n]:
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
            # if sufferages[task] < 0:
            #     raise ValueError(f"Task {task} has negative sufferage {sufferages[task]}")
        new_order = sorted(sufferages.keys(), key=sufferages.get, reverse=True)
        return new_order + queue[top_n:]


compare_funcs = {
    "EFT": lambda new, cur: new.end - cur.end,
    "EST": lambda new, cur: new.start - cur.start,
    "Quickest": lambda new, cur: (new.end - new.start) - (cur.end - cur.start),
}
insert_funcs = {}
for name, compare_func in compare_funcs.items():
    insert_funcs[f"{name}_Append"] = GreedyInsert(append_only=True, compare=compare_func)
    insert_funcs[f"{name}_Insert"] = GreedyInsert(append_only=False, compare=compare_func)
    insert_funcs[f"{name}_Append_CP"] = CriticalPathGreedyInsert(append_only=True, compare=compare_func)
    insert_funcs[f"{name}_Insert_CP"] = CriticalPathGreedyInsert(append_only=False, compare=compare_func)

initial_priority_funcs = {
    "UpwardRanking": UpwardRanking(),
    "CPoPRanking": CPoPRanking(),
    "ArbitraryTopological": ArbitraryTopological()
}

schedulers = {}
for name, insert_func in insert_funcs.items():
    schedulers[f"{name}_Sufferage"] = ParametricScheduler(
        initial_priority=ArbitraryTopological(), # Doesn't matter
        insert_task=insert_func,
        update_priority=SufferageUpdatePriority(insert_func, top_n=2)
    )
    for intial_priority_name, initial_priority_func in initial_priority_funcs.items():
        schedulers[f"{name}_{intial_priority_name}"] = ParametricScheduler(
            initial_priority=initial_priority_func,
            insert_task=insert_func,
            update_priority=NoUpdate()
        )

        for k in range(1, 4):
            schedulers[f"{name}_{intial_priority_name}_K{k}"] = ParametricKDepthScheduler(
                scheduler=schedulers[f"{name}_{intial_priority_name}"],
                k_depth=k
            )

def test_scheduler_equivalence(scheduler, base_scheduler):
    task_graphs = {
        "diamond": add_random_weights(get_diamond_dag()),
        "chain": add_random_weights(get_chain_dag()),
        "fork": add_random_weights(get_fork_dag()),
        "branching": add_random_weights(get_branching_dag(levels=3, branching_factor=2)),
    }
    network = add_random_weights(get_network())

    for task_graph_name, task_graph in task_graphs.items():
        scheduler_schedule = scheduler.schedule(network, task_graph)
        scheduler_makespan = max(
            task.end for node, tasks in scheduler_schedule.items() for task in tasks
        )
        base_scheduler_schedule = base_scheduler.schedule(network, task_graph)
        base_scheduler_makespan = max(
            task.end for node, tasks in base_scheduler_schedule.items() for task in tasks
        )
        assert np.isclose(scheduler_makespan, base_scheduler_makespan), f"Makespans not equal for {task_graph_name}. {pformat(scheduler_schedule)} {pformat(base_scheduler_schedule)}"
        logging.info(f"PASSED: Makespans are equal for  Schedulers {scheduler.__name__} and {base_scheduler.__name__} on {task_graph_name}.")
    logging.info(f"PASSED: Schedulers {scheduler.__name__} and {base_scheduler.__name__} are equivalent.")


def test():
    logging.basicConfig(level=logging.DEBUG)

    thisdir = pathlib.Path(__file__).parent.resolve()
    savedir = thisdir / "results" / "parametric" / "tests"
    if savedir.exists():
        shutil.rmtree(savedir)
    savedir.mkdir(exist_ok=True, parents=True)

    # # Uncomment to test a specific scheduler
    # scheduler_names_to_test = [
    #     "EST_Insert_CPoPRanking",
    # ]
    # for scheduler_name_to_test in scheduler_names_to_test:
    #     test_schedulers(
    #         {scheduler_name_to_test: schedulers[scheduler_name_to_test]},
    #         savedir=savedir,
    #         stop_on_error=True
    #     )

    # Test equivalence of HEFT and Parametric HEFT
    from saga.schedulers.heft import HeftScheduler
    heft = HeftScheduler()
    heft_parametric = ParametricScheduler(
        initial_priority=UpwardRanking(),
        update_priority=NoUpdate(),
        insert_task=GreedyInsert(
            append_only=False,
            compare=compare_funcs["EFT"]
        )
    )
    test_scheduler_equivalence(heft, heft_parametric)

    # Test equivalence of CPOP and Parametric CPOP
    from saga.schedulers.cpop import CpopScheduler
    cpop = CpopScheduler()
    cpop_parametric = ParametricScheduler(
        initial_priority=CPoPRanking(),
        update_priority=NoUpdate(),
        insert_task=CriticalPathGreedyInsert(
            append_only=False,
            compare=compare_funcs["EFT"]
        )
    )
    test_scheduler_equivalence(cpop, cpop_parametric)

    # Test equivalence of Suffrage and Parametric Suffrage
    from saga.schedulers.sufferage import SufferageScheduler
    etf = SufferageScheduler()
    etf_parametric = ParametricScheduler(
        initial_priority=ArbitraryTopological(),
        update_priority=SufferageUpdatePriority(insert_funcs["EFT_Insert"], top_n=2),
        insert_task=insert_funcs["EFT_Insert"]
    )
    test_scheduler_equivalence(etf, etf_parametric)
    
    # Test all schedulers
    test_schedulers(schedulers, savedir=savedir, stop_on_error=True)

def print_schedulers():
    for i, (name, scheduler) in enumerate(schedulers.items()):
        print(f"{i}: {name}")

def evaluate_scheduler(scheduler_name: str,
                       datadir: pathlib.Path,
                       resultsdir: pathlib.Path,
                       trim: int = 0,
                       overwrite: bool = False):
    datadir = datadir.resolve(strict=True)
    resultsdir = resultsdir.resolve(strict=False)
    default_datasets = [path.stem for path in datadir.glob("*.json")]
    default_datasets = ["chains", "in_trees", "out_trees"]
    scheduler = schedulers[scheduler_name]
    scheduler.name = scheduler_name
    for dataset_name in default_datasets:
        savepath = resultsdir / scheduler_name / f"{dataset_name}.csv"
        if savepath.exists() and not overwrite:
            print(f"Results already exist for {scheduler_name} on {dataset_name}. Skipping.")
            continue
        print(f"Evaluating {scheduler_name} on {dataset_name}")
        dataset = load_dataset(datadir, dataset_name)
        if trim > 0:
            dataset = TrimmedDataset(dataset, max_instances=5)
        rows = []
        for i, (network, task_graph) in enumerate(dataset):
            print(f"  Instance {i}/{len(dataset)}", end="\r")
            task_graph = standardize_task_graph(task_graph)
            t0 = time.time()
            schedule = scheduler.schedule(network, task_graph)
            dt = time.time() - t0
            makespan = max(task.end for tasks in schedule.values() for task in tasks)
            rows.append([scheduler_name, dataset_name, i, makespan, dt])
    
        df = pd.DataFrame(rows, columns=["scheduler", "dataset", "instance", "makespan", "runtime"])
        savepath.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(savepath)
        print(f"  saved results to {savepath}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, required=True, help="Directory to load the dataset from.")
    parser.add_argument("--resultsdir", type=str, required=True, help="Directory to save the results.")
    parser.add_argument("--scheduler", required=True, help="Scheduler to benchmark. Can be a number or a name.")
    parser.add_argument("--trim", type=int, default=0, help="Maximum number of instances to evaluate. Default is 0, which means no trimming.")
    args = parser.parse_args()

    if args.scheduler == "all":
        for scheduler_name in schedulers:
            evaluate_scheduler(
                scheduler_name,
                pathlib.Path(args.datadir),
                pathlib.Path(args.resultsdir),
                trim=args.trim
            )
    else:
        try:
            num = int(args.scheduler)
            if num < 0 or num >= len(schedulers):
                raise ValueError(f"Invalid scheduler number {num}. Must be between 0 and {len(schedulers) - 1}.")
            scheduler_name = list(schedulers.keys())[args.num]
        except ValueError:
            if args.scheduler not in schedulers:
                raise ValueError(f"Invalid scheduler name {args.scheduler}. Must be one of {list(schedulers.keys())}.")
            scheduler_name = args.scheduler

        evaluate_scheduler(
            scheduler_name,
            pathlib.Path(args.datadir),
            pathlib.Path(args.resultsdir),
            trim=args.trim
        )

if __name__ == "__main__":
    # test()
    # print_schedulers()
    main()