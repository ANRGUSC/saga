from functools import lru_cache
import json
import logging
import pathlib
import random
from typing import Callable, Dict, List, Optional

import networkx as nx
import numpy as np

from saga.data import AllPairsDataset, Dataset, PairsDataset
from saga.schedulers.data.random import (gen_in_trees, gen_out_trees,
                                         gen_parallel_chains,
                                         gen_random_networks)
from saga.schedulers.data.riotbench import (get_etl_task_graphs,
                                            get_fog_networks,
                                            get_predict_task_graphs,
                                            get_stats_task_graphs,
                                            get_train_task_graphs)
from saga.schedulers.data.wfcommons import (get_networks, get_real_workflows,
                                            get_workflows)

def save_dataset(savedir: pathlib.Path, dataset: Dataset) -> pathlib.Path:
    """Save the dataset to disk."""
    dataset_file = savedir.joinpath(f"{dataset.name}.json")
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Saving dataset to %s.", dataset_file)
    dataset_file.write_text(dataset.to_json(), encoding="utf-8")
    logging.info("Saved dataset to %s.", dataset_file)
    return dataset_file

@lru_cache(maxsize=None)
def load_dataset(datadir: pathlib.Path, name: str) -> Dataset:
    """Load the dataset from disk."""
    if datadir.joinpath(name).is_dir():
        return LargeDataset.from_json(
            datadir.joinpath(f"{name}.json").read_text(encoding="utf-8")
        )

    dataset_file = datadir.joinpath(f"{name}.json")
    if not dataset_file.exists():
        datasets = [path.stem for path in datadir.glob("*.json")]
        raise ValueError(f"Dataset {name} does not exist. Available datasets are {datasets}")
    text = dataset_file.read_text(encoding="utf-8")
    try:
        return AllPairsDataset.from_json(text)
    except Exception: # pylint: disable=broad-except
        return PairsDataset.from_json(text)

class LargeDataset(Dataset):
    """A dataset that is too large to fit in memory."""
    def __init__(self, datasets: Dict[pathlib.Path, int], name: str) -> None:
        super().__init__(name)
        self.datasets = datasets
        self._length = sum(datasets.values())

        self._loaded_dataset: Optional[Dataset] = None
        self._loaded_path: Optional[pathlib.Path] = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[nx.Graph, nx.DiGraph]:
        if index >= len(self):
            raise IndexError
        cum_sum = 0
        for path, length in self.datasets.items():
            if index < cum_sum + length:
                if self._loaded_path != path:
                    _dataset_json = path.read_text(encoding="utf-8")
                    try:
                        self._loaded_dataset = AllPairsDataset.from_json(_dataset_json)
                    except Exception: # pylint: disable=broad-except
                        self._loaded_dataset = PairsDataset.from_json(_dataset_json)
                    self._loaded_path = path
                return self._loaded_dataset[index - cum_sum]
            cum_sum += length
        raise IndexError

    def to_json(self, *args, **kwargs) -> str:
        """Convert the dataset to JSON."""
        return json.dumps({
            "type": "LargeDataset",
            "datasets": {
                str(path): length for path, length in self.datasets.items()
            },
            "name": self.name
        }, *args, **kwargs)

    @classmethod
    def from_json(cls, text: str, *args, **kwargs) -> "LargeDataset":
        """Load the dataset from JSON."""
        data = json.loads(text, *args, **kwargs)
        if data["type"] != "LargeDataset":
            raise ValueError("JSON is not a LargeDataset")
        return cls({
            pathlib.Path(path): length for path, length in data["datasets"].items()
        }, data["name"])

def scale_ccr(task_graph: nx.DiGraph, network: nx.DiGraph, ccr: float) -> None:
    """Scale the edge weights of the network so that the CCR is equal to the given value."""
    if ccr is None:
        return
    avg_node_speed = np.mean([
        network.nodes[node]["weight"]
        for node in network.nodes
    ])
    avg_task_cost = np.mean([
        task_graph.nodes[node]["weight"]
        for node in task_graph.nodes
    ])
    avg_data_size = np.mean([
        task_graph.edges[(src, dst)]["weight"]
        for src, dst in task_graph.edges
        if src != dst
    ])
    avg_comp_time = avg_task_cost / avg_node_speed
    link_strength = avg_data_size / (ccr * avg_comp_time)
    for src, dst in network.edges:
        if src == dst:
            continue
        network.edges[src, dst]["weight"] = link_strength

def in_trees_dataset(ccr: float = None) -> Dataset:
    """Generate the in_trees dataset."""
    num_instances = 1000
    min_levels, max_levels = 2, 4
    min_branching, max_branching = 2, 3
    min_nodes, max_nodes = 3, 5
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
        scale_ccr(task_graph, network, ccr)
        pairs.append((network, task_graph))
    return PairsDataset(pairs, name="in_trees" if ccr is None else f"in_trees_ccr_{ccr}")

def out_trees_dataset(ccr: float = None) -> Dataset:
    """Generate the out_trees dataset."""
    num_instances = 1000
    min_levels, max_levels = 2, 4
    min_branching, max_branching = 2, 3
    min_nodes, max_nodes = 3, 5
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
        scale_ccr(task_graph, network, ccr)
        pairs.append((network, task_graph))
    return PairsDataset(pairs, name="out_trees" if ccr is None else f"out_trees_ccr_{ccr}")


def chains_dataset(ccr: float = None) -> Dataset:
    """Generate the chains dataset."""
    num_instances = 1000
    min_chains, max_chains = 2, 5
    min_chain_length, max_chain_length = 2, 5
    min_nodes, max_nodes = 3, 5
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
        scale_ccr(task_graph, network, ccr)
        pairs.append((network, task_graph))
    return PairsDataset(pairs, name="chains" if ccr is None else f"chains_ccr_{ccr}")

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

def riotbench_dataset(get_task_graphs: Callable[[int], List[nx.DiGraph]],
                      name: str) -> Dataset:
    """Generate the etl dataset."""
    num_instances = 100
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

    return PairsDataset(pairs, name=name)

def run(savedir: pathlib.Path, skip_existing: bool = True):
    """Generate the datasets."""
    savedir.mkdir(parents=True, exist_ok=True)

    # Random Graphs
    if not savedir.joinpath("in_trees.json").exists() or not skip_existing:
        # in_trees_dataset()
        save_dataset(savedir, in_trees_dataset())
    if not savedir.joinpath("out_trees.json").exists() or not skip_existing:
        # out_trees_dataset()
        save_dataset(savedir, out_trees_dataset())
    if not savedir.joinpath("chains.json").exists() or not skip_existing:
        # chains_dataset()
        save_dataset(savedir, chains_dataset())

    # # Riotbench
    if not savedir.joinpath("etl.json").exists() or not skip_existing:
        # riotbench_dataset(get_etl_task_graphs, name="etl")
        save_dataset(savedir, riotbench_dataset(get_etl_task_graphs, name="etl"))
    if not savedir.joinpath("predict.json").exists() or not skip_existing:
        # riotbench_dataset(get_predict_task_graphs, name="predict")
        save_dataset(savedir, riotbench_dataset(get_predict_task_graphs, name="predict"))
    if not savedir.joinpath("stats.json").exists() or not skip_existing:
        # riotbench_dataset(get_stats_task_graphs, name="stats")
        save_dataset(savedir, riotbench_dataset(get_stats_task_graphs, name="stats"))
    if not savedir.joinpath("train.json").exists() or not skip_existing:
        # riotbench_dataset(get_train_task_graphs, name="train")
        save_dataset(savedir, riotbench_dataset(get_train_task_graphs, name="train"))

    # # Wfcommons
    for recipe_name in ["epigenomics", "montage", "cycles", "seismology", "soykb", "srasearch", "genome", "blast", "bwa"]:
        if not savedir.joinpath(f"{recipe_name}.json").exists() or not skip_existing:
            # wfcommons_dataset(recipe_name)
            save_dataset(savedir, wfcommons_dataset(recipe_name))

def run_wfcommons_ccrs(savedir: pathlib.Path, skip_existing: bool = True):
    """Generate the datasets."""
    savedir.mkdir(parents=True, exist_ok=True)

    # Wfcommons w/ different CCRs
    ccrs = [1/5, 1/2, 1, 2, 5]
    for ccr in ccrs:
        for recipe_name in ["epigenomics", "montage", "cycles", "seismology", "soykb", "srasearch", "genome", "blast", "bwa"]:
            if not savedir.joinpath(f"{recipe_name}_ccr_{ccr}.json").exists() or not skip_existing:
                # wfcommons_dataset(recipe_name, ccr=ccr, dataset_name=f"{recipe_name}_ccr_{ccr}")
                save_dataset(savedir, wfcommons_dataset(recipe_name, ccr=ccr, dataset_name=f"{recipe_name}_ccr_{ccr}"))

def prepare_ccr_datasets(savedir: pathlib.Path,
                         ccrs: List[float] = [1/5, 1/2, 1, 2, 5],
                         skip_existing: bool = True):
    """Generate the datasets."""
    savedir.mkdir(parents=True, exist_ok=True)
    dataset_names = {
        'chains': chains_dataset,
        'in_trees': in_trees_dataset,
        'out_trees': out_trees_dataset,
        'cycles': lambda ccr: wfcommons_dataset('cycles', ccr=ccr, dataset_name=f"cycles_ccr_{ccr}"),
    }
    for ccr in ccrs:
        for name in dataset_names:
            if not savedir.joinpath(f"{name}_ccr_{ccr}.json").exists() or not skip_existing:
                save_dataset(savedir, dataset_names[name](ccr=ccr))