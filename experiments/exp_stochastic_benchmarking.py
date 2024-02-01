import logging
import pathlib
import random
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from prepare_datasets import load_dataset
from saga.data import Dataset

import saga.schedulers as saga_schedulers
from saga.scheduler import Scheduler
from saga.schedulers.stochastic.determinizer import Determinizer
from saga.utils.random_variable import RandomVariable
from exp_benchmarking import get_schedulers, exclude_schedulers, TrimmedDataset

determinizers: Dict[str, Callable[[RandomVariable], float]] = {
    "mean": lambda rv: rv.mean(),
    "mean+std": lambda rv: rv.mean() + rv.std(),
    "SHEFT": lambda rv: rv.mean() + rv.std() if rv.var()/(rv.mean()**2) <= 1 else rv.mean()*(1+1/rv.std())
}

def determinize_schedulers(schedulers: List[Scheduler]) -> List[Scheduler]:
    """Determinize the schedulers.
    
    Args:
        schedulers (List[Scheduler]): The schedulers.
    
    Returns:
        List[Scheduler]: The determinized schedulers.
    """
    determinized_schedulers = []
    for scheduler in schedulers:
        for determize_name, determinize in determinizers.items():
            determinized_scheduler = Determinizer(scheduler=scheduler, determinize=determinize)
            determinized_scheduler.name = f"{scheduler.__name__}[{determize_name}]"
            determinized_schedulers.append(determinized_scheduler)
    return determinized_schedulers

def evaluate_dataset(datadir: pathlib.Path,
                     resultsdir: pathlib.Path,
                     dataset_name: str,
                     schedulers: List[Scheduler],
                     max_instances: int = 0,
                     num_jobs: int = 1,
                     overwrite: bool = False):
    """Evaluate a dataset.
    
    Args:
        datadir (pathlib.Path): The directory containing the dataset.
        resultsdir (pathlib.Path): The directory to save the results.
        dataset_name (str): The name of the dataset.
        schedulers (List[Scheduler]): The schedulers to evaluate.
        max_instances (int, optional): Maximum number of instances to evaluate. Defaults to 0 (no trimming).
        num_jobs (int, optional): The number of jobs to run in parallel. Defaults to 1.
        overwrite (bool, optional): Whether to overwrite existing results. Defaults to False.
    """
    logging.info("Evaluating dataset %s.", dataset_name)
    savepath = resultsdir.joinpath(f"{dataset_name}.csv")
    df_existing = None
    if savepath.exists():
        if not overwrite:
            # load schedulers that have already been evaluated
            df_existing = pd.read_csv(savepath, index_col=0)
            evaluated_schedulers = set(df_existing["scheduler"].unique())
            logging.info(f"Skipping {len(evaluated_schedulers)} already evaluated schedulers: {evaluated_schedulers}")
            schedulers = [scheduler for scheduler in schedulers if scheduler.__name__ not in evaluated_schedulers]
            logging.info(f"Evaluating remaining schedulers: {[scheduler.__name__ for scheduler in schedulers]}")
        else:
            logging.info("Results already exist. Skipping.")
            return
    
    dataset = load_dataset(datadir, dataset_name)
    if max_instances > 0 and len(dataset) > max_instances:
        dataset = TrimmedDataset(dataset, max_instances)
    logging.info("Loaded dataset %s.", dataset_name)
    logging.info("Running comparison for %d schedulers.", len(schedulers))
    comparison = dataset.compare(schedulers, num_jobs=num_jobs)

    logging.info("Saving results.")
    df_comp = comparison.to_df()
    if df_existing is not None:
        df_comp = pd.concat([df_existing, df_comp])
    df_comp.to_csv(savepath)
    logging.info("Saved results to %s.", savepath)

def run(datadir: pathlib.Path,
        resultsdir: pathlib.Path,
        dataset: str = None,
        num_jobs: int = 1,
        trim: int = 0,
        schedulers: List[Scheduler] = None,
        overwrite: bool = False):
    """Run the benchmarking.
    
    Args:
        datadir (pathlib.Path): The directory to save the results.
        dataset (str, optional): The name of the dataset. Defaults to None (all datasets will be evaluated).
        num_jobs (int, optional): The number of jobs to run in parallel. Defaults to 1.
        trim (int, optional): Maximum number of instances to evaluate per dataset. Defaults to 0 (no trimming).
        schedulers (List[Scheduler], optional): The schedulers to evaluate. Defaults to None (all schedulers).
        overwrite (bool, optional): Whether to overwrite existing results. Defaults to False.
    """ 
    random.seed(0) # For reproducibility
    np.random.seed(0) # For reproducibility
    resultsdir.mkdir(parents=True, exist_ok=True)

    schedulers = determinize_schedulers(schedulers if schedulers else get_schedulers())

    default_datasets = [path.stem for path in datadir.glob("*.json")]
    dataset_names = [dataset] if dataset else default_datasets
    for dataset_name in dataset_names:
        evaluate_dataset(
            datadir=datadir,
            resultsdir=resultsdir,
            dataset_name=dataset_name,
            schedulers=schedulers,
            max_instances=trim,
            num_jobs=num_jobs,
            overwrite=overwrite
        )
