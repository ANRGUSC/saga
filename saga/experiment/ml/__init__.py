from copy import deepcopy
import json
import pathlib
from typing import Dict, List
from saga.data import serialize_graph, deserialize_graph
from saga.experiment.benchmarking.parametric.components import UpwardRanking, GreedyInsert, ParametricScheduler
from saga.utils.random_graphs import add_random_weights, get_branching_dag, get_network
import networkx as nx
import os


from saga.experiment import datadir, resultsdir, outputdir

def prepare_dataset(levels: int = 3,
                    branching_factor: int = 2,
                    num_nodes: int = 10,
                    num_problems: int = 100,
                    savepath: pathlib.Path = None):
    dataset = []
    for i in range(num_problems):
        print(f"Progress: {(i+1)/num_problems*100:.2f}%" + " " * 10, end="\r")
        task_graph = add_random_weights(get_branching_dag(levels=levels, branching_factor=branching_factor))
        network = add_random_weights(get_network(num_nodes=num_nodes))

        topological_sorts = list(nx.all_topological_sorts(task_graph))
        for topological_sort in topological_sorts:
            scheduler = ParametricScheduler(
                initial_priority=lambda *_: deepcopy(topological_sort),
                insert_task=GreedyInsert(
                    append_only=False,
                    compare="EFT",
                    critical_path=False
                )
            )

            schedule = scheduler.schedule(network.copy(), task_graph.copy())
            makespan = max(task.end for tasks in schedule.values() for task in tasks)

            dataset.append(
                {
                    "instance": i,
                    "task_graph": serialize_graph(task_graph),
                    "network": serialize_graph(network),
                    "topological_sort": deepcopy(topological_sort),
                    "makespan": makespan,
                }
            )

    print("Progress: 100.00%")

    savepath.parent.mkdir(parents=True, exist_ok=True)
    savepath.write_text(json.dumps(dataset, indent=4))

def load_dataset(cache: bool = True,
                 task_graph_levels: int = 3,
                 task_graph_branching_factor: int = 2,
                 network_nodes: int = 10,
                 num_instances: int = 100) -> List[Dict]:
    """Load the dataset from disk and deserialize the graphs.
    
    Args:
        cache (bool, optional): Whether to cache the dataset. Defaults to True.
        task_graph_levels (int, optional): Number of levels in the task graph. Defaults to 3.
        task_graph_branching_factor (int, optional): Branching factor of the task graph. Defaults to 2.
        network_nodes (int, optional): Number of nodes in the network. Defaults to 10.
        num_instances (int, optional): Number of instances to generate. Defaults to 100.

    Returns:
        List[Dict]: List of problem instances, each containing a task graph, network, topological sort, and makespan
            of HEFT on the problem instance with the given topological sort.
    """
    savedir = datadir / "ml" / f"l{task_graph_levels}_b{task_graph_branching_factor}_n{network_nodes}_i{num_instances}"
    dataset_path = savedir / "data.json"
    if not dataset_path.exists() or not cache:
        print("Generating dataset...")
        prepare_dataset(
            levels=task_graph_levels,
            branching_factor=task_graph_branching_factor,
            num_nodes=network_nodes,
            num_problems=num_instances,
            savepath=dataset_path
        )

    print("Loading dataset...")
    dataset = json.loads(dataset_path.read_text())
    # deserialize graphs
    for problem_instance in dataset:
        problem_instance["task_graph"] = deserialize_graph(problem_instance["task_graph"])
        problem_instance["network"] = deserialize_graph(problem_instance["network"])
    return dataset
