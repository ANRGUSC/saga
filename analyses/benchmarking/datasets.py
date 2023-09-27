import json
import logging
import pathlib
import random
import re
from typing import Callable, Dict, List, Optional

import networkx as nx
import numpy as np

from saga.data import AllPairsDataset, Dataset, PairsDataset
from saga.schedulers.data.listvscluster import \
    get_network as get_listvscluster_network
from saga.schedulers.data.listvscluster import \
    load_task_graphs as get_listvscluster_task_graphs
from saga.schedulers.data.listvscluster import \
    get_categories as get_listvscluster_categories
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

def save_dataset(dataset: Dataset) -> pathlib.Path:
    """Save the dataset to disk."""
    dataset_file = datapath.joinpath(f"{dataset.name}.json")
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.write_text(dataset.to_json(), encoding="utf-8")
    logging.info("Saved dataset to %s.", dataset_file)
    return dataset_file

def load_dataset(name: str) -> Dataset:
    """Load the dataset from disk."""
    if datapath.joinpath(name).is_dir():
        return LargeDataset.from_json(
            datapath.joinpath(f"{name}.json").read_text(encoding="utf-8")
        )

    dataset_file = datapath.joinpath(f"{name}.json")
    if not dataset_file.exists():
        datasets = [path.stem for path in datapath.glob("*.json")]
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

def in_trees_dataset():
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
        pairs.append((network, task_graph))
    dataset = PairsDataset(pairs, name="in_trees")
    save_dataset(dataset)

def out_trees_dataset():
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
        pairs.append((network, task_graph))
    dataset = PairsDataset(pairs, name="out_trees")
    save_dataset(dataset)


def chains_dataset():
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

    dataset = PairsDataset(pairs, name=name)
    save_dataset(dataset)


def listvscluster_dataset():
    num_networks_per_task_graph = 10
    categories = get_listvscluster_categories()
    for category in categories:
        # remove puncutation from category name and replace spaces with underscores (remove duplicate spaces first)
        category_clean = re.sub(r"\s+", " ", category.lower())
        category_clean = re.sub(r"[^\w\s]", "", category_clean)
        category_clean = category_clean.replace(" ", "_")

        if datapath.joinpath(f"lvc_{category_clean}.json").exists():
            print(f"Skipping LvC Dataset: {category_clean}")
            continue

        task_graphs = get_listvscluster_task_graphs(category)

        _datasets = {}
        print(f"Generating LvC Dataset: {category_clean}")
        for i, task_graph in enumerate(task_graphs):
            pairs = []
            # gen 10 networks of different sizes ranging from 2 to graph size
            network_sizes = np.linspace(2, task_graph.number_of_nodes(), num_networks_per_task_graph, dtype=int)
            for network_size in network_sizes:
                network = get_listvscluster_network(network_size)
                pairs.append((network, task_graph))
            dataset = PairsDataset(pairs, name=f"lvc_{category_clean}/{i}")
            _datasets[save_dataset(dataset)] = num_networks_per_task_graph

        dataset = LargeDataset(_datasets, name=f"lvc_{category_clean}")
        save_dataset(dataset)

def main():
    """Generate the datasets."""
    random.seed(9281995) # For reproducibility

    # # # Random Graphs
    # in_trees_dataset()
    # out_trees_dataset()
    # chains_dataset()

    # Riotbench
    # riotbench_dataset(get_etl_task_graphs, name="etl")
    # riotbench_dataset(get_predict_task_graphs, name="predict")
    # riotbench_dataset(get_stats_task_graphs, name="stats")
    # riotbench_dataset(get_train_task_graphs, name="train")

    # Wfcommons
    # wfcommons_dataset("epigenomics")
    # wfcommons_dataset("montage")
    # wfcommons_dataset("cycles")
    # wfcommons_dataset("seismology")
    # wfcommons_dataset("soykb")
    # wfcommons_dataset("srasearch")
    # wfcommons_dataset("genome")
    # wfcommons_dataset("blast")
    # wfcommons_dataset("bwa")

    # List vs Cluster
    listvscluster_dataset()



if __name__ == "__main__":
    main()