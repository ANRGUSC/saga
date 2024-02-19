import argparse
from copy import deepcopy
from functools import lru_cache
import heapq
import logging
import os
import pathlib
from pprint import pformat
import shutil
import tempfile
import time
import fcntl

import numpy as np
import pandas as pd
from experiments.exp_benchmarking import TrimmedDataset
from experiments.prepare_datasets import load_dataset
from saga.data import Dataset
from saga.scheduler import Scheduler, Task
from saga.schedulers.parametric import IntialPriority, ScheduleType, InsertTask, ParametricScheduler
from saga.schedulers.heft import heft_rank_sort, get_insert_loc
from saga.schedulers.cpop import cpop_ranks

import networkx as nx
from typing import Any, Callable, Dict, List, Hashable, Optional, Tuple
from saga.utils.random_graphs import add_random_weights, get_branching_dag, get_chain_dag, get_diamond_dag, get_fork_dag, get_network

from saga.utils.testing import test_schedulers
from saga.utils.tools import standardize_task_graph

# Initial Priority functions
class UpwardRanking(IntialPriority):
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> List[Hashable]:
        return heft_rank_sort(network, task_graph)
    
    def serialize(self) -> Dict[str, Any]:
        return {"name": "UpwardRanking"}
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "UpwardRanking":
        return cls()

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
    
    def serialize(self) -> Dict[str, Any]:
        return {"name": "CPoPRanking"}
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "CPoPRanking":
        return cls()

class ArbitraryTopological(IntialPriority):
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> List[Hashable]:
        return list(nx.topological_sort(task_graph))
    
    def serialize(self) -> Dict[str, Any]:
        return {"name": "ArbitraryTopological"}
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "ArbitraryTopological":
        return cls()

GREEDY_INSERT_COMPARE_FUNCS = {
    "EFT": lambda new, cur: new.end - cur.end,
    "EST": lambda new, cur: new.start - cur.start,
    "Quickest": lambda new, cur: (new.end - new.start) - (cur.end - cur.start),
}
# Insert Task functions
class GreedyInsert(InsertTask):
    def __init__(self,
                 append_only: bool = False,
                 compare: str = "EFT"):
        """Initialize the GreedyInsert class.
        
        Args:
            append_only (bool, optional): Whether to only append the task to the schedule. Defaults to False.
            compare (Callable[[Task, Task], float], optional): The comparison function to use. Defaults to lambda new, cur: new.end - cur.end.
                Must be one of "EFT", "EST", or "Quickest".
        """
        self.append_only = append_only
        self.compare = compare
        self._compare = GREEDY_INSERT_COMPARE_FUNCS[compare]

    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: ScheduleType,
                 task: Hashable,
                 node: Optional[Hashable] = None) -> Task:
        best_insert_loc, best_task = None, None
        # TODO: This is a bit inefficient adds O(n) per iteration,
        #       so it doesn't affect asymptotic complexity, but it's still bad
        task_map = {
            task.name: task
            for _, tasks in schedule.items()
            for task in tasks
        }
        considered_nodes = network.nodes if node is None else [node]
        for node in considered_nodes:
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
            if best_task is None or self._compare(new_task, best_task) < 0:
                best_insert_loc, best_task = insert_loc, new_task

        schedule[best_task.node].insert(best_insert_loc, best_task)
        return best_task
    
    def serialize(self) -> Dict[str, Any]:
        return {
            "name": "GreedyInsert",
            "append_only": self.append_only,
            "compare": self.compare,
            "critical_path": False
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "GreedyInsert":
        return cls(
            append_only=data["append_only"],
            compare=data["compare"]
        )

class CriticalPathGreedyInsert(GreedyInsert):
    def __init__(self, append_only: bool = False, compare: Callable[[Task, Task], float] = lambda new, cur: new.end - cur.end):
        super().__init__(
            append_only=append_only,
            compare=compare
        )

    @lru_cache(maxsize=None)
    def cpop_ranks(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
        return cpop_ranks(network, task_graph)

    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: ScheduleType,
                 task: Hashable,
                 node: Optional[Hashable] = None) -> Task:
        # Same as GreedyInsert, but commits to scheduling nodes on the critical path to the fastest node.
        ranks = self.cpop_ranks(network, task_graph)
        critical_rank = max(ranks.values())
        critical_tasks = {
            task for task, rank in ranks.items()
            if np.isclose(rank, critical_rank)
        }
        scheduled_tasks = {
            task.name: task
            for _, tasks in schedule.items()
            for task in tasks
        }

        if task in critical_tasks and node is None:
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
            return super().__call__(network, task_graph, schedule, task, node)
        
    def serialize(self) -> Dict[str, Any]:
        return {
            "name": "CriticalPathGreedyInsert",
            "append_only": self.append_only,
            "compare": self.compare,
            "critical_path": True
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "CriticalPathGreedyInsert":
        return cls(
            append_only=data["append_only"],
            compare=data["compare"]
        )

class ParametricKDepthScheduler(Scheduler):
    def __init__(self,
                 scheduler: ParametricScheduler,
                 k_depth: int) -> None:
            super().__init__()
            self.scheduler = scheduler
            self.k_depth = k_depth
    
    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: Optional[ScheduleType] = None) -> ScheduleType:
        """Schedule the tasks on the network.
        
        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (Optional[ScheduleType]): The current schedule.

        Returns:
            Dict[Hashable, List[Task]]: A dictionary mapping nodes to a list of tasks executed on the node.
        """
        queue = self.scheduler.initial_priority(network, task_graph)
        schedule = {node: [] for node in network.nodes} if schedule is None else deepcopy(schedule)
        scheduled_tasks: Dict[Hashable, Task] = {}
        while queue:
            task_name = queue.pop(0)
            k_depth_successors = nx.single_source_shortest_path_length(
                G=task_graph,
                source=task_name,
                cutoff=self.k_depth
            )
            sub_task_graph = task_graph.subgraph(
                set(scheduled_tasks.keys()) | set(k_depth_successors.keys()) | {task_name}
            )
            
            best_node, best_makespan = None, float('inf')
            for node in network.nodes:
                sub_schedule = deepcopy(schedule) # copy the schedule
                self.scheduler.insert_task(network, task_graph, sub_schedule, task_name, node) # insert the task
                sub_schedule = self.scheduler.schedule(network, sub_task_graph, sub_schedule)
                sub_schedule_makespan = max(
                    task.end for tasks in sub_schedule.values() for task in tasks
                )
                if sub_schedule_makespan < best_makespan:
                    best_node = node
                    best_makespan = sub_schedule_makespan

            task = self.scheduler.insert_task(network, task_graph, schedule, task_name, best_node)
            scheduled_tasks[task.name] = task
            
        return schedule

    def serialize(self) -> Dict[str, Any]:
        """Return a dictionary representation of the initial priority.
        
        Returns:
            Dict[str, Any]: A dictionary representation of the initial priority.
        """
        return {
            **self.scheduler.serialize(),
            "name": "ParametricKDepthScheduler",
            "k_depth": self.k_depth
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "ParametricKDepthScheduler":
        """Return a new instance of the initial priority from the serialized data.
        
        Args:
            data (Dict[str, Any]): The serialized data.

        Returns:
            ParametricKDepthScheduler: A new instance of the initial priority.
        """
        return cls(
            scheduler=ParametricScheduler.deserialize(data),
            k_depth=data["k_depth"]
        )

class ParametricSufferageScheduler(ParametricScheduler):
    def __init__(self, scheduler: ParametricScheduler, top_n: int = 2):
        super().__init__(
            initial_priority=scheduler.initial_priority,
            insert_task=scheduler.insert_task
        )
        self.scheduler = scheduler
        self.top_n = top_n

    def serialize(self) -> Dict[str, Any]:
        return {
            **self.scheduler.serialize(),
            "name": "ParametricSufferageScheduler",
            "sufferage_top_n": self.top_n
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "ParametricSufferageScheduler":
        return cls(top_n=data["top_n"])
    
    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: Optional[ScheduleType] = None) -> ScheduleType:
        """Schedule the tasks on the network.
        
        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (Optional[ScheduleType]): The current schedule.

        Returns:
            Dict[Hashable, List[Task]]: A dictionary mapping nodes to a list of tasks executed on the node.
        """
        queue = self.initial_priority(network, task_graph)
        schedule = {node: [] for node in network.nodes} if schedule is None else deepcopy(schedule)
        scheduled_tasks: Dict[Hashable, Task] = {}
        while queue:
            ready_tasks = [
                task for task in queue
                if all(pred in scheduled_tasks for pred in task_graph.predecessors(task))
            ]
            max_sufferage_task, max_sufferage = None, -np.inf
            # print(ready_tasks)
            for task in ready_tasks[:self.top_n]:
                best_task = self.insert_task(network, task_graph, deepcopy(schedule), task)
                _network = network.copy()
                _network.nodes[best_task.node]['weight'] = 1e-9
                second_best_task = self.insert_task(_network, task_graph, deepcopy(schedule), task)
                sufferage = self.insert_task._compare(second_best_task, best_task)
                if sufferage > max_sufferage:
                    max_sufferage_task, max_sufferage = best_task, sufferage

            # print(f"Scheduling {max_sufferage_task.name} on {max_sufferage_task.node}")
            new_task = self.insert_task(network, task_graph, schedule, max_sufferage_task.name, max_sufferage_task.node)
            scheduled_tasks[new_task.name] = new_task
            queue.remove(new_task.name)

        return schedule
        
    

insert_funcs: Dict[str, GreedyInsert] = {}
for compare_func_name in GREEDY_INSERT_COMPARE_FUNCS.keys():
    insert_funcs[f"{compare_func_name}_Append"] = GreedyInsert(append_only=True, compare=compare_func_name)
    insert_funcs[f"{compare_func_name}_Insert"] = GreedyInsert(append_only=False, compare=compare_func_name)
    insert_funcs[f"{compare_func_name}_Append_CP"] = CriticalPathGreedyInsert(append_only=True, compare=compare_func_name)
    insert_funcs[f"{compare_func_name}_Insert_CP"] = CriticalPathGreedyInsert(append_only=False, compare=compare_func_name)

initial_priority_funcs = {
    "UpwardRanking": UpwardRanking(),
    "CPoPRanking": CPoPRanking(),
    "ArbitraryTopological": ArbitraryTopological()
}

schedulers: Dict[str, ParametricScheduler] = {}
for name, insert_func in insert_funcs.items():
    for intial_priority_name, initial_priority_func in initial_priority_funcs.items():
        reg_scheduler = ParametricScheduler(
            initial_priority=initial_priority_func,
            insert_task=insert_func
        )
        reg_scheduler.name = f"{name}_{intial_priority_name}"
        schedulers[reg_scheduler.name] = reg_scheduler

        sufferage_scheduler = ParametricSufferageScheduler(
            scheduler=reg_scheduler,
            top_n=None
        )
        sufferage_scheduler.name = f"{reg_scheduler.name}_Sufferage"
        schedulers[sufferage_scheduler.name] = sufferage_scheduler

        # for scheduler in [sufferage_scheduler, reg_scheduler]:
        #     for k in range(1, 4):
        #         k_depth_scheduler = ParametricKDepthScheduler(
        #             scheduler=scheduler,
        #             k_depth=k
        #         )
        #         k_depth_scheduler.name = f"{scheduler.name}_K{k}"
        #         if k == 3:
        #             schedulers[k_depth_scheduler.name] = k_depth_scheduler

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
        assert np.isclose(scheduler_makespan, base_scheduler_makespan), f"Makespans not equal for Schedulers {scheduler.__name__} and {base_scheduler.__name__} on {task_graph_name}. {pformat(scheduler_schedule)} {pformat(base_scheduler_schedule)}"
        logging.info(f"PASSED: Makespans are equal for Schedulers {scheduler.__name__} and {base_scheduler.__name__} on {task_graph_name}.")
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
        insert_task=GreedyInsert(append_only=False, compare="EFT")
    )
    test_scheduler_equivalence(heft, heft_parametric)

    # Test equivalence of CPOP and Parametric CPOP
    from saga.schedulers.cpop import CpopScheduler
    cpop = CpopScheduler()
    cpop_parametric = ParametricScheduler(
        initial_priority=CPoPRanking(),
        insert_task=CriticalPathGreedyInsert(append_only=False, compare="EFT")
    )
    test_scheduler_equivalence(cpop, cpop_parametric)

    # Test equivalence of Suffrage and Parametric Suffrage
    from saga.schedulers.sufferage import SufferageScheduler
    etf = SufferageScheduler()
    etf_parametric = ParametricSufferageScheduler(
        scheduler=ParametricScheduler(
            initial_priority=ArbitraryTopological(),
            insert_task=insert_funcs["EFT_Append"]
        ),
        top_n=None
    )
    test_scheduler_equivalence(etf, etf_parametric)
    
    # # Test all schedulers
    test_schedulers(schedulers, savedir=savedir, stop_on_error=True)

def print_schedulers():
    for i, (name, scheduler) in enumerate(schedulers.items(), start=1):
        print(f"{i}: {name}")

def print_datasets(datadir: pathlib.Path):
    for i, dataset_path in enumerate(datadir.glob("*.json"), start=1):
        dataset = load_dataset(datadir, dataset_path.stem)
        print(f"{i}: {dataset_path.stem} ({len(dataset)} instances)")

def append_df_to_csv_with_lock(df: pd.DataFrame, savepath: pathlib.Path):
    lock_path = savepath.with_suffix(f"{savepath.suffix}.lock")
    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if savepath.exists():
            df.to_csv(savepath, mode='a', header=False, index=False)
        else:
            df.to_csv(savepath, index=False)
        fcntl.flock(lock_file, fcntl.LOCK_UN)

DEFAULT_DATASETS = [
    f"{name}_ccr_{ccr}"
    for name in ['chains', 'in_trees', 'out_trees']
    for ccr in [0.2, 0.5, 1, 2, 5]
]
def evaluate_instance(scheduler: Scheduler,
                      datadir: pathlib.Path,
                      dataset_name: str,
                      instance_num: int,
                      savepath: pathlib.Path):
    datadir = datadir.resolve(strict=True)
    dataset = load_dataset(datadir, dataset_name)
    network, task_graph = dataset[instance_num]
    # savepath = resultsdir / scheduler.__name__ / f"{dataset_name}_{instance_num}.csv"
    print(f"Evaluating {scheduler.__name__} on {dataset_name} instance {instance_num}")
    task_graph = standardize_task_graph(task_graph)
    t0 = time.time()
    schedule = scheduler.schedule(network, task_graph)
    dt = time.time() - t0
    makespan = max(task.end for tasks in schedule.values() for task in tasks)
    df = pd.DataFrame(
        [[scheduler.__name__, dataset_name, instance_num, makespan, dt]],
        columns=["scheduler", "dataset", "instance", "makespan", "runtime"]
    )
    savepath.parent.mkdir(exist_ok=True, parents=True)
    append_df_to_csv_with_lock(df, savepath)
    print(f"  saved results to {savepath}")

def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")
    run_subparser = subparsers.add_parser("run", help="Run the benchmarking.")
    run_subparser.add_argument("--datadir", type=str, required=True, help="Directory to load the dataset from.")
    run_subparser.add_argument("--out", type=str, required=True, help="Directory to save the results.")
    run_subparser.add_argument("--trim", type=int, default=0, help="Maximum number of instances to evaluate. Default is 0, which means no trimming.")
    run_subparser.add_argument("--batch", type=int, required=True, help="Batch number to run.")

    list_subparser = subparsers.add_parser("list", help="List the available schedulers.")

    test_subparser = subparsers.add_parser("test", help="Run the tests.")
    args = parser.parse_args()

    if args.command == "test":
        test()
    elif args.command == "list":
        print_schedulers()
    else:
        datadir = pathlib.Path(args.datadir)
        datasets = {
            dataset_name: len(load_dataset(datadir, datadir.joinpath(f"{dataset_name}.json").stem))
            for dataset_name in DEFAULT_DATASETS
        }
        instances = [
            (scheduler, dataset_name, instance_num)
            for scheduler in schedulers.values()
            for dataset_name, dataset_size in datasets.items()
            for instance_num in range(dataset_size if args.trim <= 0 else min(dataset_size, args.trim))
        ]

        # group to 5000 roughly equal sized sets of instances
        num_batches = 5000
        batches = [[] for _ in range(num_batches)]
        for i, instance in enumerate(instances):
            batches[i % num_batches].append(instance)

        if args.batch < 0 or args.batch >= num_batches:
            raise ValueError(f"Invalid batch number {args.batch}. Must be between 0 and {num_batches-1} for this trim value ({args.trim}).")
        batch = batches[args.batch]
        print(f"Batch {args.batch} has {len(batch)} instances.")
        for scheduler, dataset_name, instance_num in batch:
            evaluate_instance(
                scheduler,
                datadir,
                dataset_name,
                instance_num,
                pathlib.Path(args.out)
            )

def test_filelock():
    import multiprocessing
    import pathlib

    thisdir = pathlib.Path(__file__).resolve().parent

    start = time.time() + 5

    def write_to_file(filename: pathlib.Path, text: str):
        # sleep until the start time
        time.sleep(max(0, start - time.time()))
        for i in range(100):
            rows = [
                [0, 1, 2, 3, 4, 5]
            ]
            df = pd.DataFrame(rows, columns=["a", "b", "c", "d", "e", "f"])
            append_df_to_csv_with_lock(df, filename)
            # df.to_csv(filename, mode='a', header=False, index=False)

    savepath = thisdir / "results" / "test.csv"
    if savepath.exists():
        savepath.unlink()
    savepath.parent.mkdir(exist_ok=True, parents=True)
    # create 100 processes to write to the file
    processes = [
        multiprocessing.Process(target=write_to_file, args=(savepath, f"Process {i}"))
        for i in range(1000)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    print("Done.")


if __name__ == "__main__":
    # test()
    # print_schedulers()
    # print_datasets(pathlib.Path(__file__).parent / "datasets" / "benchmarking")
    # main()
    test_filelock()
