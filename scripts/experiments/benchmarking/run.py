from typing import List, Optional, Set, Tuple, Tuple
import logging
import pathlib
import pandas as pd

from saga.schedulers.data import Dataset
from saga import Scheduler

from common import datadir, saga_schedulers, resultsdir

def evaluate_dataset(resultsdir: pathlib.Path,
                     dataset_name: str,
                     schedulers: Optional[List[Scheduler]] = None,
                     overwrite: bool = False):
    """Evaluate a dataset.
    
    Args:
        datadir (pathlib.Path): The directory containing the dataset.
        resultsdir (pathlib.Path): The directory to save the results.
        dataset_name (str): The name of the dataset.
        schedulers (Optional[List[Scheduler]], optional): The schedulers to evaluate. Defaults to None (all schedulers).
        overwrite (bool, optional): Whether to overwrite existing results. Defaults to False.
    """
    savepath = resultsdir.joinpath(f"{dataset_name}.csv")
    savepath.parent.mkdir(exist_ok=True, parents=True)
    finished: Set[Tuple[str, str, str]] = set()
    if savepath.exists() and not overwrite:
        logging.info("Results for dataset %s already exist at %s. Skipping.", dataset_name, savepath)
        finished_df = pd.read_csv(savepath)
        for _, row in finished_df.iterrows():
            finished.add((row["Dataset"], row["Instance"], row["Scheduler"]))
    
    dataset = Dataset(name=dataset_name)
    logging.info("Dataset %s has %d instances.", dataset_name, dataset.size)
    for instance in dataset.iter_instances():
        logging.info("Instance: %s", instance.name)
        for scheduler in schedulers or saga_schedulers.values():
            logging.info("  Scheduler: %s", scheduler.__class__.__name__)
            if (dataset.name, instance.name, scheduler.__class__.__name__) in finished:
                logging.info("    Already finished. Skipping.")
                continue
            schedule = scheduler.schedule(
                network=instance.network,
                task_graph=instance.task_graph
            )
            makespan = schedule.makespan
            # append results to csv
            result_df = pd.DataFrame([{
                "Dataset": dataset.name,
                "Instance": instance.name,
                "Scheduler": scheduler.__class__.__name__,
                "Makespan": makespan
            }])
            if savepath.exists():
                result_df.to_csv(savepath, mode="a", header=False, index=False)
            else:
                result_df.to_csv(savepath, index=False)

def main():    
    datasets = [path.name for path in datadir.iterdir() if path.is_dir()]
    for dataset in datasets:
        logging.info("Evaluating dataset %s", dataset)
        evaluate_dataset(resultsdir, dataset, overwrite=False)  # Evaluate each dataset

if __name__ == "__main__":
    main()
