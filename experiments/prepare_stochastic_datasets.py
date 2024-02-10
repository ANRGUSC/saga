import json
import logging
import pathlib
import random
from typing import Callable, Dict, List, Optional

import networkx as nx
import numpy as np

from saga.data import AllPairsDataset, Dataset, PairsDataset
from saga.schedulers.stochastic.data.random import (
    gen_in_trees,
    gen_out_trees,
    gen_parallel_chains,
    gen_random_networks
)
from saga.schedulers.stochastic.data.wfcommons import (
    get_networks,
    get_real_workflows,
    get_workflows
)

from prepare_datasets import save_dataset, load_dataset, LargeDataset

def in_trees_dataset() -> Dataset:
    """Generate the in_trees dataset."""
    num_instances = 100
    min_levels, max_levels = 2, 4
    min_branching, max_branching = 2, 3
    min_nodes, max_nodes = 3, 5
    pairs = []
    for _ in range(num_instances):
        print(f"Generating in-trees {len(pairs)+1}/{num_instances}")
        network = gen_random_networks(
            num=1,
            num_nodes=random.randint(min_nodes, max_nodes)
        )[0]
        task_graph = gen_in_trees(
            num=1,
            num_levels=random.randint(min_levels, max_levels),
            branching_factor=random.randint(min_branching, max_branching)
        )[0]
        pairs.append((network, task_graph))
    return PairsDataset(pairs, name="in_trees")

def out_trees_dataset() -> Dataset:
    """Generate the out_trees dataset."""
    num_instances = 100
    min_levels, max_levels = 2, 4
    min_branching, max_branching = 2, 3
    min_nodes, max_nodes = 3, 5
    pairs = []
    for _ in range(num_instances):
        print(f"Generating out_trees {len(pairs)+1}/{num_instances}")
        network = gen_random_networks(
            num=1,
            num_nodes=random.randint(min_nodes, max_nodes)
        )[0]
        task_graph = gen_out_trees(
            num=1,
            num_levels=random.randint(min_levels, max_levels),
            branching_factor=random.randint(min_branching, max_branching)
        )[0]
        pairs.append((network, task_graph))
    return PairsDataset(pairs, name="out_trees")

def chains_dataset() -> Dataset:
    """Generate the chains dataset."""
    num_instances = 1
    min_chains, max_chains = 2, 5
    min_chain_length, max_chain_length = 2, 5
    min_nodes, max_nodes = 3, 5
    pairs = []
    for _ in range(num_instances):
        print(f"Generating chains {len(pairs)+1}/{num_instances}")
        network = gen_random_networks(
            num=1,
            num_nodes=random.randint(min_nodes, max_nodes)
        )[0]
        task_graph = gen_parallel_chains(
            num=1,
            num_chains=random.randint(min_chains, max_chains),
            chain_length=random.randint(min_chain_length, max_chain_length)
        )[0]
        pairs.append((network, task_graph))
    return PairsDataset(pairs, name="chains")

def wfcommons_dataset(recipe_name: str,
                      ccr: float = 1.0,
                      num_instances: int = 100,
                      dataset_name: Optional[str] = None) -> None:
    """Generate the wfcommons dataset.

    Args:
        recipe_name: The name of the recipe to generate the dataset for.
        ccr: The communication to computation ratio - the ratio of the average
            edge weight to the average node weight. This is used to scale the
            edge weights so that the CCR is equal to the given value for the
            real workflows the dataset is based on.
        num_instances: The number of instances to generate.
        dataset_name: The name of the dataset to generate. If None, the recipe
            name is used.
    """
    workflows = get_real_workflows(recipe_name)
    mean_task_weight = np.mean([
        workflow.nodes[node]["weight"]
        for workflow in workflows
        for node in workflow.nodes
    ])
    mean_edge_weight = np.mean([
        workflow.edges[edge]["weight"]
        for workflow in workflows
        for edge in workflow.edges
    ])
    mean_comp_time = mean_task_weight # since avg comp time is 1 by construction
    link_strength = mean_edge_weight / (ccr * mean_comp_time)
    logging.info("Link strength for %s CCR=%s: %s", recipe_name, ccr, link_strength)

    networks = get_networks(num=100, cloud_name='chameleon', network_speed=link_strength)
    pairs = []
    for i in range(num_instances):
        print(f"Generating {recipe_name} {i+1}/{num_instances}")
        network = networks[i]
        task_graph = get_workflows(num=1, recipe_name=recipe_name)[0]
        pairs.append((network, task_graph))

    return PairsDataset(pairs, name=dataset_name or recipe_name)

def run(savedir: pathlib.Path, skip_existing: bool = True):
    """Generate the datasets."""
    random.seed(0) # For reproducibility
    np.random.seed(0)
    savedir.mkdir(parents=True, exist_ok=True)

    print("Generating datasets", skip_existing, not savedir.joinpath("in_trees.json").exists() or not skip_existing)

    # Random Graphs
    if not savedir.joinpath("in_trees.json").exists() or not skip_existing:
        save_dataset(savedir, in_trees_dataset())
    
    if not savedir.joinpath("out_trees.json").exists() or not skip_existing:
        save_dataset(savedir, out_trees_dataset())

    if not savedir.joinpath("chains.json").exists() or not skip_existing:
        save_dataset(savedir, chains_dataset())

    # # # Riotbench
    # if not savedir.joinpath("etl.json").exists() or not skip_existing:
    #     # riotbench_dataset(get_etl_task_graphs, name="etl")
    #     save_dataset(savedir, riotbench_dataset(get_etl_task_graphs, name="etl"))
    # if not savedir.joinpath("predict.json").exists() or not skip_existing:
    #     # riotbench_dataset(get_predict_task_graphs, name="predict")
    #     save_dataset(savedir, riotbench_dataset(get_predict_task_graphs, name="predict"))
    # if not savedir.joinpath("stats.json").exists() or not skip_existing:
    #     # riotbench_dataset(get_stats_task_graphs, name="stats")
    #     save_dataset(savedir, riotbench_dataset(get_stats_task_graphs, name="stats"))
    # if not savedir.joinpath("train.json").exists() or not skip_existing:
    #     # riotbench_dataset(get_train_task_graphs, name="train")
    #     save_dataset(savedir, riotbench_dataset(get_train_task_graphs, name="train"))

    # # # Wfcommons
    # for recipe_name in ["epigenomics", "montage", "cycles", "seismology", "soykb", "srasearch", "genome", "blast", "bwa"]:
    #     if not savedir.joinpath(f"{recipe_name}.json").exists() or not skip_existing:
    #         print(f"Generating {recipe_name}")
    #         save_dataset(savedir, wfcommons_dataset(recipe_name))
