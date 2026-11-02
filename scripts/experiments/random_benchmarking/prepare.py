import random
from multiprocessing import Pool
from typing import Callable, List, Optional, Tuple

from common import exclude_datasets, num_processors

from saga import TaskGraph
from saga.schedulers.data import Dataset, ProblemInstance
from saga.schedulers.data.random import (
    gen_in_trees,
    gen_out_trees,
    gen_parallel_chains,
    gen_random_networks,
)


def in_trees_dataset(ccr: Optional[float] = None, overwrite: bool = False) -> Dataset:
    """Generate the in_trees dataset.

    Args:
        ccr: The communication to computation ratio.
        overwrite: Whether to overwrite existing instances.

    Returns:
        Dataset: The generated dataset.
    """
    dataset_name = f"in_trees_ccr_{ccr}" if ccr is not None else "in_trees"
    dataset = Dataset(name=dataset_name)

    num_instances = 100
    min_levels, max_levels = 2, 4
    min_branching, max_branching = 2, 3
    min_nodes, max_nodes = 3, 5
    existing_instances = set(dataset.instances) if not overwrite else set()
    for i in range(num_instances):
        intance_name = f"{dataset_name}_{i}"
        if intance_name in existing_instances:
            continue
        network = gen_random_networks(
            num=1, num_nodes=random.randint(min_nodes, max_nodes)
        )[0]
        task_graph = gen_in_trees(
            num=1,
            num_levels=random.randint(min_levels, max_levels),
            branching_factor=random.randint(min_branching, max_branching),
        )[0]
        if ccr is not None:
            network = network.scale_to_ccr(task_graph, ccr)
        dataset.save_instance(
            ProblemInstance(name=intance_name, network=network, task_graph=task_graph)
        )
    return dataset


def out_trees_dataset(ccr: Optional[float] = None, overwrite: bool = False) -> Dataset:
    """Generate the out_trees dataset.

    Args:
        ccr: The communication to computation ratio.
        overwrite: Whether to overwrite existing instances.

    Returns:
        Dataset: The generated dataset.
    """
    dataset_name = f"out_trees_ccr_{ccr}" if ccr is not None else "out_trees"
    dataset = Dataset(name=dataset_name)
    num_instances = 100
    min_levels, max_levels = 2, 4
    min_branching, max_branching = 2, 3
    min_nodes, max_nodes = 3, 5
    existing_instances = set(dataset.instances) if not overwrite else set()
    for i in range(num_instances):
        instance_name = f"{dataset_name}_{i}"
        if instance_name in existing_instances:
            continue
        network = gen_random_networks(
            num=1, num_nodes=random.randint(min_nodes, max_nodes)
        )[0]
        task_graph = gen_out_trees(
            num=1,
            num_levels=random.randint(min_levels, max_levels),
            branching_factor=random.randint(min_branching, max_branching),
        )[0]
        if ccr is not None:
            network = network.scale_to_ccr(task_graph, ccr)
        dataset.save_instance(
            ProblemInstance(name=instance_name, network=network, task_graph=task_graph)
        )
    return dataset


def chains_dataset(ccr: Optional[float] = None, overwrite: bool = False) -> Dataset:
    """Generate the chains dataset.

    Args:
        ccr: The communication to computation ratio.
        overwrite: Whether to overwrite existing instances.

    Returns:
        Dataset: The generated dataset.
    """
    dataset_name = f"chains_ccr_{ccr}" if ccr is not None else "chains"
    dataset = Dataset(name=dataset_name)

    num_instances = 100
    min_chains, max_chains = 2, 5
    min_chain_length, max_chain_length = 2, 5
    min_nodes, max_nodes = 3, 5
    existing_instances = set(dataset.instances) if not overwrite else set()
    for i in range(num_instances):
        instance_name = f"{dataset_name}_{i}"
        if instance_name in existing_instances:
            continue
        network = gen_random_networks(
            num=1, num_nodes=random.randint(min_nodes, max_nodes)
        )[0]
        task_graph = gen_parallel_chains(
            num=1,
            num_chains=random.randint(min_chains, max_chains),
            chain_length=random.randint(min_chain_length, max_chain_length),
        )[0]
        if ccr is not None:
            network = network.scale_to_ccr(task_graph, ccr)
        dataset.save_instance(
            ProblemInstance(name=instance_name, network=network, task_graph=task_graph)
        )
    return dataset


def _prepare_dataset_task(task: Tuple[str, dict]) -> Tuple[str, int]:
    """Worker function to prepare a single dataset.

    Args:
        task: Tuple of (task_type, kwargs) where task_type identifies the dataset
              and kwargs are the arguments to pass to the dataset function.

    Returns:
        Tuple of (dataset_name, dataset_size).
    """
    task_type, kwargs = task

    if task_type == "in_trees":
        ds = in_trees_dataset(**kwargs)
    elif task_type == "out_trees":
        ds = out_trees_dataset(**kwargs)
    elif task_type == "chains":
        ds = chains_dataset(**kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    print(f"Generated dataset {ds.name} with {ds.size} instances.")
    return ds.name, ds.size


def prepare_datasets(overwrite: bool = False, num_workers: int = num_processors):
    """Generate the datasets using multiprocessing.

    Args:
        overwrite: Whether to overwrite existing instances.
        num_workers: Number of worker processes.
    """
    tasks: List[Tuple[str, dict]] = []

    # Random Graphs
    tasks.append(("in_trees", {"overwrite": overwrite}))
    tasks.append(("out_trees", {"overwrite": overwrite}))
    tasks.append(("chains", {"overwrite": overwrite}))

    # Riotbench
    tasks.append(("riotbench_etl", {"overwrite": overwrite}))
    tasks.append(("riotbench_predict", {"overwrite": overwrite}))
    tasks.append(("riotbench_stats", {"overwrite": overwrite}))
    tasks.append(("riotbench_train", {"overwrite": overwrite}))

    # Wfcommons
    for recipe_name in [
        "epigenomics",
        "montage",
        "cycles",
        "seismology",
        "soykb",
        "srasearch",
        "genome",
        "blast",
        "bwa",
    ]:
        tasks.append((f"wfcommons_{recipe_name}", {"overwrite": overwrite}))

    # remove any excluded datasets
    # tasks = [task for task in tasks if task[0] not in exclude_datasets]
    tasks_not_excluded = []
    for dataset_name, kwargs in tasks:
        if any(
            excluded_dataset in dataset_name for excluded_dataset in exclude_datasets
        ):
            print(f"Skipping excluded dataset {dataset_name}")
            continue
        tasks_not_excluded.append((dataset_name, kwargs))

    with Pool(processes=num_workers) as pool:
        results = pool.map(_prepare_dataset_task, tasks_not_excluded)

    print(f"\nPrepared {len(results)} datasets.")


def main():
    prepare_datasets(overwrite=False)


if __name__ == "__main__":
    main()
