import logging
import pathlib
import pandas as pd

from datasets import load_dataset
from saga.data import Dataset

import saga.schedulers as saga_schedulers
from saga.scheduler import Scheduler
from saga.utils.tools import validate_simple_schedule

logging.basicConfig(level=logging.DEBUG)
thisdir = pathlib.Path(__file__).parent.absolute()
exclude_schedulers = [ # exclude optimal schedulers
    saga_schedulers.BruteForceScheduler,
    saga_schedulers.SMTScheduler,
]

def main():
    """Run the benchmarking."""
    schedulers = []
    for item in saga_schedulers.__dict__.values():
        if (isinstance(item, type) and issubclass(item, Scheduler) and item is not Scheduler):
            if item not in exclude_schedulers:
                try:
                    schedulers.append(item())
                except TypeError:
                    logging.warning("Could not instantiate %s with default arguments.", item.__name__)

    # schedulers = [saga_schedulers.CpopScheduler()]

    datasets: dict[str, Dataset] = {}
    for dataset_path in thisdir.joinpath(".datasets").glob("*.json"):
        logging.info("Loading dataset %s.", dataset_path.stem)
        datasets[dataset_path.stem] = load_dataset(dataset_path.stem)

    datasets = {k: datasets[k] for k in ["in_trees", "out_trees", "parallel_chains", "blast_chameleon", "riotbench"]}

    df_data, skip_datasets = None, set()
    if thisdir.joinpath("results.csv").exists():
        df_data = pd.read_csv(thisdir.joinpath("results.csv"), index_col=0)
        skip_datasets = set(df_data["dataset"].unique())

    for dataset_name, dataset in datasets.items():
        if dataset_name in skip_datasets:
            logging.info("Skipping dataset %s.", dataset_name)
            continue
        logging.info("Evaluating dataset %s.", dataset_name)
        comparison = dataset.compare(schedulers)

        _df_data = comparison.to_df()
        _df_data["dataset"] = dataset_name
        if df_data is None:
            df_data = _df_data
        else:
            # remove where scheduler and dataset columns are the same
            df_data = pd.concat([df_data, _df_data], ignore_index=True)
        df_data.to_csv(thisdir.joinpath("results.csv"))

if __name__ == "__main__":
    main()
