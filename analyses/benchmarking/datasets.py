import logging
import pathlib
import random
from typing import Callable, List

import networkx as nx

from saga.data import Dataset, PairsDataset, AllPairsDataset
from saga.schedulers.data.random import (gen_in_trees, gen_out_trees,
                                         gen_parallel_chains,
                                         gen_random_networks)
from saga.schedulers.data.riotbench import (get_etl_task_graphs,
                                            get_fog_networks,
                                            get_predict_task_graphs,
                                            get_stats_task_graphs,
                                            get_train_task_graphs)
from saga.schedulers.data.wfcommons import get_networks, get_workflows

logging.basicConfig(level=logging.INFO)
thisdir = pathlib.Path(__file__).parent.absolute()
datapath = thisdir.joinpath(".datasets")

def save_dataset(dataset: Dataset):
    """Save the dataset to disk."""
    datapath.mkdir(parents=True, exist_ok=True)
    dataset_file = datapath.joinpath(f"{dataset.name}.json")
    dataset_file.write_text(dataset.to_json(), encoding="utf-8")
    logging.info("Saved dataset to %s.", dataset_file)

def load_dataset(name: str) -> Dataset:
    """Load the dataset from disk."""
    dataset_file = datapath.joinpath(f"{name}.json")
    if not dataset_file.exists():
        datasets = [path.stem for path in datapath.glob("*.json")]
        raise ValueError(f"Dataset {name} does not exist. Available datasets are {datasets}")
    text = dataset_file.read_text(encoding="utf-8")
    try:
        return AllPairsDataset.from_json(text)
    except Exception: # pylint: disable=broad-except
        return PairsDataset.from_json(text)

def in_trees_dataset():
    """Generate the in_trees dataset."""
    num_instances = 2500
    min_levels, max_levels = 3, 5
    min_branching, max_branching = 2, 4
    min_nodes, max_nodes = 3, 10
    pairs = []
    for _ in range(num_instances):
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
    dataset = PairsDataset(pairs, name="in_trees")
    save_dataset(dataset)

def out_trees_dataset():
    """Generate the out_trees dataset."""
    num_instances = 2500
    min_levels, max_levels = 3, 5
    min_branching, max_branching = 2, 4
    min_nodes, max_nodes = 3, 10
    pairs = []
    for _ in range(num_instances):
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
    dataset = PairsDataset(pairs, name="out_trees")
    save_dataset(dataset)


def chains_dataset():
    """Generate the chains dataset."""
    num_instances = 2500
    min_chains, max_chains = 2, 5
    min_chain_length, max_chain_length = 2, 5
    min_nodes, max_nodes = 3, 10
    pairs = []
    for _ in range(num_instances):
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
    dataset = PairsDataset(pairs, name="chains")
    save_dataset(dataset)

def wfcommons_dataset(recipe_name: str):
    """Generate the blast dataset."""
    num_instances = 100
    pairs = []
    networks = get_networks(num=100, cloud_name='chameleon')
    for i in range(num_instances):
        print(f"Generating {recipe_name} {i+1}/{num_instances}")
        network = networks[i]
        task_graph = get_workflows(num=1, recipe_name=recipe_name)[0]
        pairs.append((network, task_graph))

    dataset = PairsDataset(pairs, name=recipe_name)
    save_dataset(dataset)

def riotbench_dataset(get_task_graphs: Callable[[int], List[nx.DiGraph]],
                      name: str):
    """Generate the etl dataset."""
    num_instances = 1000
    pairs = []
    min_edge_nodes, max_edge_nodes = 75, 125
    min_fog_nodes, max_fog_nodes = 3, 7
    min_cloud_nodes, max_cloud_nodes = 1, 10
    for i in range(num_instances):
        network = get_fog_networks(
            num=1,
            num_edges_nodes=random.randint(min_edge_nodes, max_edge_nodes),
            num_fog_nodes=random.randint(min_fog_nodes, max_fog_nodes),
            num_cloud_nodes=random.randint(min_cloud_nodes, max_cloud_nodes)
        )[0]
        task_graph = get_task_graphs(num=1)[0]
        pairs.append((network, task_graph))

    dataset = PairsDataset(pairs, name=name)
    save_dataset(dataset)


def main():
    """Generate the datasets."""
    random.seed(9281995) # For reproducibility

    # # Random Graphs
    # in_trees_dataset()
    # out_trees_dataset()
    # chains_dataset()

    # # Riotbench
    # riotbench_dataset(get_etl_task_graphs, name="etl")
    # riotbench_dataset(get_predict_task_graphs, name="predict")
    # riotbench_dataset(get_stats_task_graphs, name="stats")
    # riotbench_dataset(get_train_task_graphs, name="train")

    # Wfcommons
    # wfcommons_dataset("epigenomics")
    # wfcommons_dataset("montage")
    # wfcommons_dataset("cycles")
    wfcommons_dataset("seismology")
    # wfcommons_dataset("soykb")
    # wfcommons_dataset("srasearch")
    # wfcommons_dataset("genome")
    # wfcommons_dataset("blast")
    # wfcommons_dataset("bwa")



if __name__ == "__main__":
    main()