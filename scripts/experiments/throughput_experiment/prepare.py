import random
from multiprocessing import Pool
from typing import Callable, List, Optional, Tuple

from common import datadir, num_processors

from saga import TaskGraph
from saga.schedulers.data import Dataset, ProblemInstance
from saga.schedulers.data.random import (
    gen_in_trees,
    gen_out_trees,
    gen_parallel_chains,
    gen_random_networks,
)
from saga.schedulers.data.riotbench import (
    get_etl_task_graphs,
    get_fog_networks,
    get_predict_task_graphs,
    get_stats_task_graphs,
    get_train_task_graphs,
)
from saga.schedulers.data.wfcommons import get_networks, get_workflows


def in_trees_dataset(ccr: Optional[float] = None, overwrite: bool = False) -> Dataset:
    dataset_name = f"in_trees_ccr_{ccr}" if ccr is not None else "in_trees"
    dataset = Dataset(name=dataset_name)
    num_instances = 100
    existing = set(dataset.instances) if not overwrite else set()
    for i in range(num_instances):
        name = f"{dataset_name}_{i}"
        if name in existing:
            continue
        network = gen_random_networks(num=1, num_nodes=random.randint(3, 5))[0]
        task_graph = gen_in_trees(
            num=1,
            num_levels=random.randint(2, 4),
            branching_factor=random.randint(2, 3),
        )[0]
        if ccr is not None:
            network = network.scale_to_ccr(task_graph, ccr)
        dataset.save_instance(ProblemInstance(name=name, network=network, task_graph=task_graph))
    return dataset


def out_trees_dataset(ccr: Optional[float] = None, overwrite: bool = False) -> Dataset:
    dataset_name = f"out_trees_ccr_{ccr}" if ccr is not None else "out_trees"
    dataset = Dataset(name=dataset_name)
    num_instances = 100
    existing = set(dataset.instances) if not overwrite else set()
    for i in range(num_instances):
        name = f"{dataset_name}_{i}"
        if name in existing:
            continue
        network = gen_random_networks(num=1, num_nodes=random.randint(3, 5))[0]
        task_graph = gen_out_trees(
            num=1,
            num_levels=random.randint(2, 4),
            branching_factor=random.randint(2, 3),
        )[0]
        if ccr is not None:
            network = network.scale_to_ccr(task_graph, ccr)
        dataset.save_instance(ProblemInstance(name=name, network=network, task_graph=task_graph))
    return dataset


def chains_dataset(ccr: Optional[float] = None, overwrite: bool = False) -> Dataset:
    dataset_name = f"chains_ccr_{ccr}" if ccr is not None else "chains"
    dataset = Dataset(name=dataset_name)
    num_instances = 100
    existing = set(dataset.instances) if not overwrite else set()
    for i in range(num_instances):
        name = f"{dataset_name}_{i}"
        if name in existing:
            continue
        network = gen_random_networks(num=1, num_nodes=random.randint(3, 5))[0]
        task_graph = gen_parallel_chains(
            num=1,
            num_chains=random.randint(2, 5),
            chain_length=random.randint(2, 5),
        )[0]
        if ccr is not None:
            network = network.scale_to_ccr(task_graph, ccr)
        dataset.save_instance(ProblemInstance(name=name, network=network, task_graph=task_graph))
    return dataset


def wfcommons_dataset(
    recipe_name: str,
    ccr: float = 1.0,
    num_instances: int = 100,
    dataset_name: Optional[str] = None,
    overwrite: bool = False,
) -> Dataset:
    dataset_name = dataset_name or recipe_name
    dataset = Dataset(name=dataset_name)
    existing = set(dataset.instances) if not overwrite else set()
    instance_names = {f"{dataset_name}_{i}" for i in range(num_instances)}
    new_instances = instance_names - existing
    if not new_instances:
        return dataset
    networks = get_networks(num=len(new_instances), cloud_name="chameleon")
    workflows = get_workflows(num=len(new_instances), recipe_name=recipe_name)
    for i, name in enumerate(new_instances):
        network = networks[i].scale_to_ccr(workflows[i], ccr)
        dataset.save_instance(ProblemInstance(name=name, network=network, task_graph=workflows[i]))
    return dataset


def riotbench_dataset(
    get_task_graphs: Callable[[int], List[TaskGraph]],
    name: str,
    num_instances: int = 100,
    overwrite: bool = False,
) -> Dataset:
    dataset = Dataset(name=name)
    existing = set(dataset.instances) if not overwrite else set()
    instance_names = {f"{name}_{i}" for i in range(num_instances)}
    new_instances = instance_names - existing
    networks = get_fog_networks(num=len(new_instances))
    task_graphs = get_task_graphs(len(new_instances))
    for i, inst_name in enumerate(new_instances):
        dataset.save_instance(
            ProblemInstance(name=inst_name, network=networks[i], task_graph=task_graphs[i])
        )
    return dataset


def _prepare_dataset_task(task: Tuple[str, dict]) -> Tuple[str, int]:
    task_type, kwargs = task
    if task_type == "in_trees":
        ds = in_trees_dataset(**kwargs)
    elif task_type == "out_trees":
        ds = out_trees_dataset(**kwargs)
    elif task_type == "chains":
        ds = chains_dataset(**kwargs)
    elif task_type == "riotbench_etl":
        ds = riotbench_dataset(get_etl_task_graphs, name="etl", **kwargs)
    elif task_type == "riotbench_predict":
        ds = riotbench_dataset(get_predict_task_graphs, name="predict", **kwargs)
    elif task_type == "riotbench_stats":
        ds = riotbench_dataset(get_stats_task_graphs, name="stats", **kwargs)
    elif task_type == "riotbench_train":
        ds = riotbench_dataset(get_train_task_graphs, name="train", **kwargs)
    elif task_type.startswith("wfcommons_"):
        recipe_name = task_type.replace("wfcommons_", "")
        ds = wfcommons_dataset(recipe_name, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    print(f"Generated dataset {ds.name} with {ds.size} instances.")
    return ds.name, ds.size


def prepare_datasets(overwrite: bool = False, num_workers: int = num_processors) -> None:
    datadir.mkdir(exist_ok=True, parents=True)
    tasks: List[Tuple[str, dict]] = [
        ("in_trees",        {"overwrite": overwrite}),
        ("out_trees",       {"overwrite": overwrite}),
        ("chains",          {"overwrite": overwrite}),
        ("riotbench_etl",   {"overwrite": overwrite}),
        ("riotbench_predict", {"overwrite": overwrite}),
        ("riotbench_stats", {"overwrite": overwrite}),
        ("riotbench_train", {"overwrite": overwrite}),
        ("wfcommons_epigenomics", {"overwrite": overwrite}),
        ("wfcommons_montage",     {"overwrite": overwrite}),
        ("wfcommons_cycles",      {"overwrite": overwrite}),
        ("wfcommons_seismology",  {"overwrite": overwrite}),
        ("wfcommons_soykb",       {"overwrite": overwrite}),
        ("wfcommons_srasearch",   {"overwrite": overwrite}),
        ("wfcommons_genome",      {"overwrite": overwrite}),
        ("wfcommons_blast",       {"overwrite": overwrite}),
        ("wfcommons_bwa",         {"overwrite": overwrite}),
    ]
    with Pool(processes=num_workers) as pool:
        results = pool.map(_prepare_dataset_task, tasks)
    print(f"\nPrepared {len(results)} datasets.")


def main():
    prepare_datasets(overwrite=False)


if __name__ == "__main__":
    main()
