import argparse
import logging
import pathlib
import random

from datasets import load_dataset
from joblib import Parallel, delayed
from saga.data import Dataset

import saga.schedulers as saga_schedulers
from saga.scheduler import Scheduler

logging.basicConfig(level=logging.DEBUG)
thisdir = pathlib.Path(__file__).parent.absolute()
results_dir = thisdir.joinpath("results")
results_dir.mkdir(parents=True, exist_ok=True)
exclude_schedulers = [ # exclude optimal schedulers
    saga_schedulers.BruteForceScheduler,
    saga_schedulers.SMTScheduler,
]

all_datasets = [
    path.stem for path in thisdir.joinpath(".datasets").glob("*.json")
]

def get_schedulers() -> list[Scheduler]:
    schedulers = []
    for item in saga_schedulers.__dict__.values():
        if (isinstance(item, type) and issubclass(item, Scheduler) and item is not Scheduler):
            if item not in exclude_schedulers:
                try:
                    schedulers.append(item())
                except TypeError:
                    logging.warning("Could not instantiate %s with default arguments.", item.__name__)
    return schedulers

class TrimmedDataset(Dataset):
    def __init__(self, dataset: Dataset, max_instances: int):
        super().__init__(dataset.name)
        self.dataset = dataset
        self.max_instances = max_instances

    def __len__(self):
        return min(len(self.dataset), self.max_instances)
    
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        return self.dataset[index]

def evaluate_dataset(dataset_name: str, max_instances: int = 0, num_jobs: int = 1):
    """Evaluate a dataset."""
    logging.info("Evaluating dataset %s.", dataset_name)
    dataset = load_dataset(dataset_name)
    if max_instances > 0 and len(dataset) > max_instances:
        dataset = TrimmedDataset(dataset, max_instances)
    logging.info("Loaded dataset %s.", dataset_name)
    schedulers = get_schedulers()
    logging.info("Running comparison.")
    comparison = dataset.compare(schedulers, num_jobs=num_jobs)

    logging.info("Saving results.")
    df_comp = comparison.to_df()
    savepath = results_dir.joinpath(f"{dataset_name}.csv")
    df_comp.to_csv(savepath)
    logging.info("Saved results to %s.", savepath)

def get_parser() -> argparse.ArgumentParser:
    """Get the parser."""
    parser = argparse.ArgumentParser(description="Evaluate a dataset.")
    parser.add_argument("datasets", nargs="+", help="The datasets to evaluate.", choices=['all', *all_datasets])
    parser.add_argument("--num-jobs", type=int, default=1, help="The number of jobs to run in parallel. Defaults to the number of cores on the machine.")
    parser.add_argument("--trim", type=int, default=0, help="Maximum number of instances to evaluate per dataset. Defaults to 0 (no trimming).")
    return parser

def main():
    """Run the benchmarking."""
    random.seed(9281995) # For reproducibility

    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    dataset_names = args.datasets if 'all' not in args.datasets else all_datasets

    # Evaluate datasets in parallel, redirect stdout/stderr to logs/<dataset_name>.log
    logdir = thisdir.joinpath("logs")
    logdir.mkdir(parents=True, exist_ok=True)
    for dataset_name in dataset_names:
        evaluate_dataset(dataset_name, args.trim, num_jobs=args.num_jobs)


if __name__ == "__main__":
    main()
