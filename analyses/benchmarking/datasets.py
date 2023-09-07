import logging
import pathlib

from saga.data import AllPairsDataset, Dataset, PairsDataset
from saga.schedulers.data.random import (gen_in_trees, gen_out_trees,
                                         gen_parallel_chains,
                                         gen_random_networks)
from saga.schedulers.data.riotbench import (get_etl_task_graphs,
                                            get_fog_networks,
                                            get_predict_task_graphs,
                                            get_stats_task_graphs,
                                            get_train_task_graphs)
from saga.schedulers.data.wfcommons import get_networks, get_workflows, recipes

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
    try:
        return AllPairsDataset.from_json(dataset_file.read_text(encoding="utf-8"))
    except Exception:
        return PairsDataset.from_json(dataset_file.read_text(encoding="utf-8"))

def main():
    """Generate the datasets."""
    # num_networks = 10
    # num_task_graphs = 10
    # networks = gen_random_networks(num=num_networks, num_nodes=5)
    # logging.info("Generated %d networks.", len(networks))

    # out_trees = gen_out_trees(num=num_task_graphs, num_levels=3, branching_factor=2)
    # save_dataset(Dataset.from_networks_and_task_graphs(networks, out_trees, name="out_trees"))

    # in_trees = gen_in_trees(num=num_task_graphs, num_levels=3, branching_factor=2)
    # save_dataset(Dataset.from_networks_and_task_graphs(networks, in_trees, name="in_trees"))

    # parallel_chains = gen_parallel_chains(num=num_task_graphs, num_chains=2, chain_length=3)
    # save_dataset(Dataset.from_networks_and_task_graphs(networks, parallel_chains, name="parallel_chains"))

    # # WFCommons
    # for recipe_name in recipes:
    #     task_graphs = get_workflows(num=20, recipe_name=recipe_name)
    #     cloud_networks = get_networks(num=20, cloud_name='chameleon')
    #     save_dataset(Dataset.from_pairs(zip(cloud_networks, task_graphs), name=f"{recipe_name}_chameleon"))

    # RiotBench
    fog_networks = get_fog_networks(num=20)
    riot_task_graphs = [
        *get_etl_task_graphs(num=20),
        *get_predict_task_graphs(num=20),
        *get_stats_task_graphs(num=20),
        *get_train_task_graphs(num=20)
    ]
    save_dataset(Dataset.from_networks_and_task_graphs(fog_networks, riot_task_graphs, name="riotbench"))


if __name__ == "__main__":
    main()