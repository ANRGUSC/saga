import json
import pathlib
from typing import Dict, List, Tuple
from prepare_datasets import load_dataset
from saga.data import Dataset, serialize_graph, deserialize_graph
import numpy as np
import random
import networkx as nx

random.seed(0)
np.random.seed(0)

def shuffle_datasets(datadir: pathlib.Path,
                     batches: int,
                     savedir: pathlib.Path,
                     trim: int = None):
    all_datasets: Dict[str, Dataset] = {}
    for dataset_path in datadir.glob("*.json"):
        all_datasets[dataset_path.stem] = load_dataset(datadir, dataset_path.stem)

    all_datasets = [
        (name, i, network, task_graph)
        for name, dataset in all_datasets.items()
        for i, (network, task_graph) in enumerate(dataset[:trim])
    ]

    # shuffle datasets
    random.shuffle(all_datasets)

    # save shuffled datasets
    savedir.mkdir(parents=True, exist_ok=True)
    for i in range(batches):
        batch_datasets = all_datasets[i::batches]
        print(f"Batch {i}: {len(batch_datasets)} datasets")

        batch_datasets = [
            (name, i, serialize_graph(network), serialize_graph(task_graph))
            for name, i, network, task_graph in batch_datasets
        ]
        savedir.joinpath(f"batch_{i}.json").write_text(json.dumps(batch_datasets))

def load_batch(datadir: pathlib.Path, batch: int) -> List[Tuple[str, int, nx.Graph, nx.DiGraph]]:
    return [
        (name, i, deserialize_graph(network), deserialize_graph(task_graph))
        for name, i, network, task_graph in json.loads(datadir.joinpath(f"batch_{batch}.json").read_text())
    ]

if __name__ == "__main__":
    shuffle_datasets(
        datadir=pathlib.Path("datasets/parametric_benchmarking"),
        batches=500,
        savedir=pathlib.Path("datasets/shuffled_parametric_benchmarking"),
        trim=100
    )

    batch_3 = load_batch(pathlib.Path("datasets/shuffled_parametric_benchmarking"), 3)
    print(len(batch_3))
        

    