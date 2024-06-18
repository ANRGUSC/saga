import logging
import pathlib
import random
from typing import Callable, Dict, Hashable, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from saga.experiment.benchmarking.prepare import load_dataset

from saga.scheduler import Scheduler, Task
from saga.schedulers import HeftScheduler, FastestNodeScheduler, CpopScheduler
from saga.schedulers.stochastic.determinizer import Determinizer
from saga.utils.random_variable import RandomVariable
from saga.experiment.benchmarking import get_schedulers, TrimmedDataset

determinizers: Dict[str, Callable[[RandomVariable], float]] = {
    "mean": lambda rv: rv.mean(),
    "mean+std": lambda rv: rv.mean() + rv.std(),
    "SHEFT": lambda rv: rv.mean() + rv.std() if rv.var()/(rv.mean()**2) <= 1 else rv.mean()*(1+1/rv.std())
}

def determinize_schedulers(schedulers: List[Scheduler]) -> List[Scheduler]:
    """Determinize the schedulers.
    
    Args:
        schedulers (List[Scheduler]): The schedulers.
    
    Returns:
        List[Scheduler]: The determinized schedulers.
    """
    determinized_schedulers = []
    for scheduler in schedulers:
        for determize_name, determinize in determinizers.items():
            determinized_scheduler = Determinizer(scheduler=scheduler, determinize=determinize)
            determinized_scheduler.name = f"{scheduler.__name__}[{determize_name}]"
            determinized_schedulers.append(determinized_scheduler)
    return determinized_schedulers

def evaluate_dataset(datadir: pathlib.Path,
                     resultsdir: pathlib.Path,
                     dataset_name: str,
                     schedulers: List[Scheduler],
                     max_instances: int = 0,
                     num_jobs: int = 1,
                     overwrite: bool = False):
    """Evaluate a dataset.
    
    Args:
        datadir (pathlib.Path): The directory containing the dataset.
        resultsdir (pathlib.Path): The directory to save the results.
        dataset_name (str): The name of the dataset.
        schedulers (List[Scheduler]): The schedulers to evaluate.
        max_instances (int, optional): Maximum number of instances to evaluate. Defaults to 0 (no trimming).
        num_jobs (int, optional): The number of jobs to run in parallel. Defaults to 1.
        overwrite (bool, optional): Whether to overwrite existing results. Defaults to False.
    """
    logging.info("Evaluating dataset %s.", dataset_name)
    savepath = resultsdir.joinpath(f"{dataset_name}.csv")
    df_existing = None
    if savepath.exists():
        if not overwrite:
            # load schedulers that have already been evaluated
            df_existing = pd.read_csv(savepath, index_col=0)
            evaluated_schedulers = set(df_existing["scheduler"].unique())
            logging.info(f"Skipping {len(evaluated_schedulers)} already evaluated schedulers: {evaluated_schedulers}")
            schedulers = [scheduler for scheduler in schedulers if scheduler.__name__ not in evaluated_schedulers]
            logging.info(f"Evaluating remaining schedulers: {[scheduler.__name__ for scheduler in schedulers]}")
        else:
            logging.info("Results already exist. Skipping.")
            return
    
    dataset = load_dataset(datadir, dataset_name)
    if max_instances > 0 and len(dataset) > max_instances:
        dataset = TrimmedDataset(dataset, max_instances)
    logging.info("Loaded dataset %s.", dataset_name)
    logging.info("Running comparison for %d schedulers.", len(schedulers))
    comparison = dataset.compare(schedulers, num_jobs=num_jobs)

    logging.info("Saving results.")
    df_comp = comparison.to_df()
    if df_existing is not None:
        df_comp = pd.concat([df_existing, df_comp])
    df_comp.to_csv(savepath)
    logging.info("Saved results to %s.", savepath)


def determinize_solution(network: nx.Graph,
                         task_graph: nx.DiGraph,
                         schedule: Dict[Hashable, List[Task]]) -> Tuple[nx.Graph, nx.DiGraph, Dict[Hashable, List[Task]]]:
    """Determinize the problem instance and solution.

    Args:
        network (nx.Graph): The network.
        task_graph (nx.DiGraph): The task graph.
        schedule (Dict[Hashable, List[Task]]): A schedule mapping nodes to a list of tasks.

    Returns:
        Tuple[nx.Graph, nx.DiGraph, Dict[Hashable, List[Task]]]: The determinized network, task graph, and schedule.
    """
    # determinize network
    new_network = nx.Graph()
    for node, node_data in network.nodes.items():
        weight = node_data["weight"]
        if isinstance(weight, RandomVariable):
            new_network.add_node(node, weight=max(1e-9, weight.sample()))
        else:

            new_network.add_node(node, weight=weight)

    for edge, edge_data in network.edges.items():
        weight = edge_data["weight"]
        if isinstance(weight, RandomVariable):
            new_network.add_edge(edge[0], edge[1], weight=max(1e-9, weight.sample()))
        else:
            new_network.add_edge(edge[0], edge[1], weight=weight)

    # determinize task graph
    new_task_graph = nx.DiGraph()
    for task, task_data in task_graph.nodes.items():
        weight = task_data["weight"]
        if isinstance(weight, RandomVariable):
            new_task_graph.add_node(task, weight=max(1e-9, weight.sample()))
        else:
            new_task_graph.add_node(task, weight=weight)

    for dep, dep_data in task_graph.edges.items():
        weight = dep_data["weight"]
        if isinstance(weight, RandomVariable):
            new_task_graph.add_edge(dep[0], dep[1], weight=max(1e-9, weight.sample()))
        else:
            new_task_graph.add_edge(dep[0], dep[1], weight=weight)

    task_map: Dict[str, Task] = {}
    for node, tasks in schedule.items():
        for task in tasks:
            task_map[task.name] = task

    task_order: List[Task] = sorted(task_map.values(), key=lambda task: task.start)
    new_schedule: Dict[Hashable, List[Task]] = {node: [] for node in new_network.nodes}
    new_task_map: Dict[str, Task] = {}
    for task in task_order:
        node = task.node
        # compute time to get data from dependencies
        data_available_time = max(
            [
                new_task_map[dep[0]].end + new_task_graph.edges[dep]["weight"] / new_network.edges[new_task_map[dep[0]].node, node]["weight"]
                for dep in new_task_graph.in_edges(task.name)
            ],
            default=0
        )
        start_time = data_available_time if not new_schedule[node] else max(data_available_time, new_schedule[node][-1].end)
        new_task = Task(node, task.name, start_time, start_time + new_task_graph.nodes[task.name]["weight"] / new_network.nodes[node]["weight"])
        new_schedule[node].append(new_task)
        new_task_map[task.name] = new_task

    return new_network, new_task_graph, new_schedule

def run(datadir: pathlib.Path,
        resultsdir: pathlib.Path,
        dataset: str = None,
        num_jobs: int = 1,
        trim: int = 0,
        schedulers: List[Scheduler] = None,
        num_samples: int = 100):
    """Run the benchmarking.
    
    Args:
        datadir (pathlib.Path): The directory to save the results.
        dataset (str, optional): The name of the dataset. Defaults to None (all datasets will be evaluated).
        num_jobs (int, optional): The number of jobs to run in parallel. Defaults to 1.
        trim (int, optional): Maximum number of instances to evaluate per dataset. Defaults to 0 (no trimming).
        schedulers (List[Scheduler], optional): The schedulers to evaluate. Defaults to None (all schedulers).
    """ 
    resultsdir.mkdir(parents=True, exist_ok=True)

    schedulers = determinize_schedulers(schedulers if schedulers else get_schedulers())

    default_datasets = [path.stem for path in datadir.glob("*.json")]
    dataset_names = [dataset] if dataset else default_datasets

    rows = []
    for dataset_name in dataset_names:
        _dataset = load_dataset(datadir, dataset_name)
        for scheduler in schedulers:
            for instance_num, (network, task_graph) in enumerate(_dataset):
                print("Evaluating", dataset_name, scheduler.__name__, instance_num)
                schedule = scheduler.schedule(network, task_graph)
                pred_makespan = max([task.end for tasks in schedule.values() for task in tasks])
                # print(f"  Predicted makespan: {pred_makespan}")
                for _sample in range(num_samples):
                    determinized_network, determinized_task_graph, determinized_schedule = determinize_solution(network, task_graph, schedule)
                    makespan = max([task.end for tasks in determinized_schedule.values() for task in tasks])
                    rows.append([dataset_name, instance_num, scheduler.__name__, _sample, makespan])
                    # print(f"  Sample {_sample} makespan: {makespan}")
            
            df = pd.DataFrame(rows, columns=["dataset", "instance", "scheduler", "sample", "makespan"])
            df.to_csv(resultsdir.joinpath(f"results.csv"))
            print("Added results for", dataset_name, scheduler.__name__)


def main():
    thisdir = pathlib.Path(__file__).resolve().parent
    datadir = thisdir.joinpath("datasets/stochastic_benchmarking")
    resultsdir = thisdir.joinpath("results/stochastic_benchmarking")
    run(
        datadir,
        resultsdir,
        schedulers=[
            FastestNodeScheduler(),
            HeftScheduler(),
            CpopScheduler(),
        ]
    )

if __name__ == "__main__":
    main()