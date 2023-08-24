import json
import logging
from pprint import pformat
import saga.schedulers as saga_schedulers
from saga.schedulers.base import Scheduler
from saga.data import Dataset
import pathlib
from saga.schedulers.data.random import (gen_in_trees, gen_out_trees,
                                         gen_parallel_chains,
                                         gen_random_networks)

logging.basicConfig(level=logging.INFO)
thisdir = pathlib.Path(__file__).parent.absolute()
exclude_schedulers = [ # exclude schedulers that are optimal
    saga_schedulers.BruteForceScheduler,
    saga_schedulers.SMTScheduler,
]

def main():
    """Run the benchmarking."""
    num_networks = 100
    num_task_graphs = 100

    schedulers = []
    for item in saga_schedulers.__dict__.values():
        if (isinstance(item, type) and issubclass(item, Scheduler) and item is not Scheduler):
            if item not in exclude_schedulers:
                try:
                    schedulers.append(item())
                except TypeError:
                    logging.warning("Could not instantiate %s with default arguments.", item.__name__)

    networks = gen_random_networks(num=num_networks, num_nodes=5)
    logging.info("Generated %d networks.", len(networks))

    out_trees = gen_out_trees(num=num_task_graphs, num_levels=3, branching_factor=2)
    logging.info("Generated %d out trees.", len(out_trees))

    in_trees = gen_in_trees(num=num_task_graphs, num_levels=3, branching_factor=2)
    logging.info("Generated %d in trees.", len(in_trees))

    parallel_chains = gen_parallel_chains(num=num_task_graphs, num_chains=2, chain_length=3)
    logging.info("Generated %d parallel chains.", len(parallel_chains))

    datasets = {
        "out_trees": Dataset.from_networks_and_task_graphs(networks, out_trees),
        "in_trees": Dataset.from_networks_and_task_graphs(networks, in_trees),
        "parallel_chains": Dataset.from_networks_and_task_graphs(networks, parallel_chains),
    }
    logging.info("Generated %d datasets.", len(datasets))

    results = {}
    for dataset_name, dataset in datasets.items():
        comparison = dataset.compare(schedulers)
        stats = comparison.stats()
        results[dataset_name] = stats

    thisdir.joinpath("results.json").write_text(json.dumps(results, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
