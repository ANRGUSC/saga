import random
from typing import Callable, List, Optional

from saga import TaskGraph
from saga.schedulers.data import Dataset, ProblemInstance
from saga.schedulers.data.random import (gen_in_trees, gen_out_trees,
                                         gen_parallel_chains,
                                         gen_random_networks)
from saga.schedulers.data.riotbench import (get_etl_task_graphs,
                                            get_fog_networks,
                                            get_predict_task_graphs,
                                            get_stats_task_graphs,
                                            get_train_task_graphs)
from saga.schedulers.data.wfcommons import (get_networks,
                                            get_workflows)

from common import datadir

def in_trees_dataset(ccr: Optional[float] = None,
                     overwrite: bool = False) -> Dataset:
    """Generate the in_trees dataset.
    
    Args:
        ccr: The communication to computation ratio.
        overwrite: Whether to overwrite existing instances.

    Returns:
        Dataset: The generated dataset.
    """
    dataset_name = f"in_trees_ccr_{ccr}" if ccr is not None else "in_trees"
    dataset = Dataset(name=dataset_name)

    num_instances = 1000
    min_levels, max_levels = 2, 4
    min_branching, max_branching = 2, 3
    min_nodes, max_nodes = 3, 5
    existing_instances = set(dataset.instances) if not overwrite else set()
    for i in range(num_instances):
        intance_name = f"{dataset_name}_{i}"
        if intance_name in existing_instances:
            continue
        network = gen_random_networks(
            num=1,
            num_nodes=random.randint(min_nodes, max_nodes)
        )[0]
        task_graph = gen_in_trees(
            num=1,
            num_levels=random.randint(min_levels, max_levels),
            branching_factor=random.randint(min_branching, max_branching)
        )[0]
        if ccr is not None:
            network = network.scale_to_ccr(task_graph, ccr)
        dataset.save_instance(
            ProblemInstance(
                name=intance_name,
                network=network,
                task_graph=task_graph
            )
        )
    return dataset
        
def out_trees_dataset(ccr: Optional[float] = None,
                      overwrite: bool = False) -> Dataset:
    """Generate the out_trees dataset.
    
    Args:
        ccr: The communication to computation ratio.
        overwrite: Whether to overwrite existing instances.
    
    Returns:
        Dataset: The generated dataset.
    """
    dataset_name = f"out_trees_ccr_{ccr}" if ccr is not None else "out_trees"
    dataset = Dataset(name=dataset_name)
    num_instances = 1000
    min_levels, max_levels = 2, 4
    min_branching, max_branching = 2, 3
    min_nodes, max_nodes = 3, 5
    existing_instances = set(dataset.instances) if not overwrite else set()
    for i in range(num_instances):
        instance_name = f"{dataset_name}_{i}"
        if instance_name in existing_instances:
            continue
        network = gen_random_networks(
            num=1,
            num_nodes=random.randint(min_nodes, max_nodes)
        )[0]
        task_graph = gen_out_trees(
            num=1,
            num_levels=random.randint(min_levels, max_levels),
            branching_factor=random.randint(min_branching, max_branching)
        )[0]
        if ccr is not None:
            network = network.scale_to_ccr(task_graph, ccr)
        dataset.save_instance(
            ProblemInstance(
                name=instance_name,
                network=network,
                task_graph=task_graph
            )
        )
    return dataset

def chains_dataset(ccr: Optional[float] = None,
                   overwrite: bool = False) -> Dataset:
    """Generate the chains dataset.
    
    Args:
        ccr: The communication to computation ratio.
        overwrite: Whether to overwrite existing instances.

    Returns:
        Dataset: The generated dataset.
    """
    dataset_name = f"chains_ccr_{ccr}" if ccr is not None else "chains"
    dataset = Dataset(name=dataset_name)

    num_instances = 1000
    min_chains, max_chains = 2, 5
    min_chain_length, max_chain_length = 2, 5
    min_nodes, max_nodes = 3, 5
    existing_instances = set(dataset.instances) if not overwrite else set()
    for i in range(num_instances):
        instance_name = f"{dataset_name}_{i}"
        if instance_name in existing_instances:
            continue
        network = gen_random_networks(
            num=1,
            num_nodes=random.randint(min_nodes, max_nodes)
        )[0]
        task_graph = gen_parallel_chains(
            num=1,
            num_chains=random.randint(min_chains, max_chains),
            chain_length=random.randint(min_chain_length, max_chain_length)
        )[0]
        if ccr is not None:
            network = network.scale_to_ccr(task_graph, ccr)
        dataset.save_instance(
            ProblemInstance(
                name=instance_name,
                network=network,
                task_graph=task_graph
            )
        )
    return dataset

def wfcommons_dataset(recipe_name: str,
                      ccr: float = 1.0,
                      num_instances: int = 100,
                      dataset_name: Optional[str] = None,
                      overwrite: bool = False) -> Dataset:
    """Generate the wfcommons dataset.

    Args:
        recipe_name: The name of the recipe to generate the dataset for.
        ccr: The communication to computation ratio.
        num_instances: The number of instances to generate.
        dataset_name: The name of the dataset to generate. If None, the recipe
            name is used.
        overwrite: Whether to overwrite existing instances.
    """
    dataset_name = dataset_name or recipe_name
    dataset = Dataset(name=dataset_name)
    existing_instances = set(dataset.instances) if not overwrite else set()
    instance_names = {f"{dataset_name}_{i}" for i in range(num_instances)}
    new_instances = instance_names - existing_instances
    if not new_instances:
        return dataset
    networks = get_networks(num=len(new_instances), cloud_name='chameleon')
    workflows = get_workflows(num=len(new_instances), recipe_name=recipe_name)
    for i, instance_name in enumerate(new_instances):
        network = networks[i]
        task_graph = workflows[i]
        if ccr is not None:
            network = network.scale_to_ccr(task_graph, ccr)
        dataset.save_instance(
            ProblemInstance(
                name=instance_name,
                network=network,
                task_graph=task_graph
            )
        )
    return dataset

def riotbench_dataset(get_task_graphs: Callable[[int], List[TaskGraph]],
                      name: str,
                      num_instances: int = 100,
                      overwrite: bool = False) -> Dataset:
    """Generate a riotbench dataset.

    Args:
        get_task_graphs: A function that returns a list of task graphs.
        name: The name of the dataset.
        num_instances: The number of instances to generate.
        overwrite: Whether to overwrite existing instances.

    Returns:
        Dataset: The generated dataset.
    """
    dataset = Dataset(name=name)
    existing_instances = set(dataset.instances) if not overwrite else set()
    isntance_names = {f"{name}_{i}" for i in range(num_instances)}
    new_instances = isntance_names - existing_instances
    networks = get_fog_networks(num=len(new_instances))
    task_graphs = get_task_graphs(len(new_instances))
    for i, instance_name in enumerate(new_instances):
        dataset.save_instance(
            ProblemInstance(
                name=instance_name,
                network=networks[i],
                task_graph=task_graphs[i]
            )
        )
    return dataset

def prepare_datasets(overwrite: bool = False):
    """Generate the datasets.
    
    Args:
        overwrite: Whether to overwrite existing instances.
    """
    # Random Graphs
    ds_intrees = in_trees_dataset(overwrite=overwrite)
    print(f"Generated dataset {ds_intrees.name} with {ds_intrees.size} instances.")
    ds_out_trees = out_trees_dataset(overwrite=overwrite)
    print(f"Generated dataset {ds_out_trees.name} with {ds_out_trees.size} instances.")
    ds_chains = chains_dataset(overwrite=overwrite)
    print(f"Generated dataset {ds_chains.name} with {ds_chains.size} instances.")

    # Riotbench
    ds_etl = riotbench_dataset(get_etl_task_graphs, name="etl", overwrite=overwrite)
    print(f"Generated dataset {ds_etl.name} with {ds_etl.size} instances.")
    ds_predict = riotbench_dataset(get_predict_task_graphs, name="predict", overwrite=overwrite)
    print(f"Generated dataset {ds_predict.name} with {ds_predict.size} instances.")
    ds_stats = riotbench_dataset(get_stats_task_graphs, name="stats", overwrite=overwrite)
    print(f"Generated dataset {ds_stats.name} with {ds_stats.size} instances.")
    ds_train = riotbench_dataset(get_train_task_graphs, name="train", overwrite=overwrite)
    print(f"Generated dataset {ds_train.name} with {ds_train.size} instances.")

    # Wfcommons
    for recipe_name in ["epigenomics", "montage", "cycles", "seismology", "soykb", "srasearch", "genome", "blast", "bwa"]:
        ds = wfcommons_dataset(recipe_name, overwrite=overwrite)
        print(f"Generated dataset {ds.name} with {ds.size} instances.")

def prepare_wfcommons_ccr_datasets(overwrite: bool = False):
    """Generate the wfcommons datasets with varying CCRs.
    
    Args:
        overwrite: Whether to overwrite existing instances.
    """
    ccrs = [1/5, 1/2, 1, 2, 5]
    dataset_names = [
        'epigenomics',
        'montage',
        'cycles',
        'seismology',
        'soykb',
        'srasearch',
        'genome',
        'blast',
        'bwa'
    ]
    for ccr in ccrs:
        for name in dataset_names:
            wfcommons_dataset(
                recipe_name=name,
                ccr=ccr,
                dataset_name=f"{name}_ccr_{ccr}",
                overwrite=overwrite
            )

def prepare_ccr_datasets(ccrs: List[float] = [1/5, 1/2, 1, 2, 5],
                         overwrite: bool = False):
    """Generate the datasets."""
    dataset_names = {
        'chains': chains_dataset,
        'in_trees': in_trees_dataset,
        'out_trees': out_trees_dataset,
        'cycles': lambda ccr: wfcommons_dataset('cycles', ccr=ccr, dataset_name=f"cycles_ccr_{ccr}"),
    }
    for ccr in ccrs:
        for name in dataset_names:
            dataset_names[name](
                ccr=ccr,
                overwrite=overwrite
            )

def main():
    prepare_datasets(overwrite=False)

if __name__ == "__main__":
    main()