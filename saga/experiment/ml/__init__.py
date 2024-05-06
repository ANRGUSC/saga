from copy import deepcopy
import json
from typing import Dict, List
from saga.data import serialize_graph, deserialize_graph
from saga.experiment.benchmarking.parametric.components import UpwardRanking, GreedyInsert, ParametricScheduler
from saga.utils.random_graphs import add_random_weights, get_branching_dag, get_network
import networkx as nx
import os


from saga.experiment import datadir, resultsdir, outputdir

def prepare_dataset():
    LEVELS = 3
    BRANCHING_FACTOR = 2
    NUM_NODES = 10
    NUM_PROBLEMS = 100

    dataset = []
    for i in range(NUM_PROBLEMS):
        print(f"Progress: {i+1/NUM_PROBLEMS*100:.2f}%" + " " * 10, end="\r")
        task_graph = add_random_weights(get_branching_dag(levels=LEVELS, branching_factor=BRANCHING_FACTOR))
        network = add_random_weights(get_network(num_nodes=NUM_NODES))

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

    savedir = datadir / "ml"
    savedir.mkdir(exist_ok=True, parents=True)
    dataset_path = savedir / "data.json"
    dataset_path.write_text(json.dumps(dataset, indent=4))

def load_dataset(cache: bool = True) -> List[Dict]:
    """Load the dataset from disk and deserialize the graphs.
    
    Args:
        cache (bool, optional): Whether to cache the dataset. Defaults to True.

    Returns:
        List[Dict]: List of problem instances, each containing a task graph, network, topological sort, and makespan
            of HEFT on the problem instance with the given topological sort.
    """
    savedir = datadir / "ml"
    dataset_path = savedir / "data.json"
    if not dataset_path.exists() or not cache:
        print("Dataset not found. Generating dataset...")
        prepare_dataset()

    print("Loading dataset...")
    dataset = json.loads(dataset_path.read_text())
    # deserialize graphs
    for problem_instance in dataset:
        problem_instance["task_graph"] = deserialize_graph(problem_instance["task_graph"])
        problem_instance["network"] = deserialize_graph(problem_instance["network"])
    return dataset
