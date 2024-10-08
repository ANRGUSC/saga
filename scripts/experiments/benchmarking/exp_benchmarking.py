import logging
import pathlib
from typing import List, Optional

from saga.data import Dataset
from saga.schedulers import (
    BILScheduler, CpopScheduler, DuplexScheduler, ETFScheduler, FCPScheduler,
    FLBScheduler, FastestNodeScheduler, GDLScheduler, HeftScheduler,
    MCTScheduler, METScheduler, MaxMinScheduler, MinMinScheduler,
    OLBScheduler, WBAScheduler
)
from saga.scheduler import Scheduler

from prepare import load_dataset, prepare_datasets
from scripts.experiments.benchmarking.post_benchmarking import run_analysis

thisdir = pathlib.Path(__file__).parent.resolve()

exclude_schedulers = []
saga_schedulers = {
    # Schedulers included in benchmarking results for the paper
    # "Comparing Task Graph Scheduling Algorithms: An Adversarial Approach"
    # https://arxiv.org/abs/2403.07120
    "BIL": BILScheduler,
    "CPoP": CpopScheduler,
    "Duplex": DuplexScheduler,
    "ETF": ETFScheduler,
    "FCP": FCPScheduler,
    "FLB": FLBScheduler,
    "FastestNode": FastestNodeScheduler,
    "GDL": GDLScheduler,
    "HEFT": HeftScheduler,
    "MCT": MCTScheduler,
    "MET": METScheduler,
    "MaxMin": MaxMinScheduler,
    "MinMin": MinMinScheduler,
    "OLB": OLBScheduler,
    "WBA": WBAScheduler
}

def get_schedulers() -> List[Scheduler]:
    """Get a list of all schedulers.
    
    Returns:
        List[Scheduler]: list of schedulers
    """
    schedulers = []
    for item in saga_schedulers.values():
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

def evaluate_dataset(datadir: pathlib.Path,
                     resultsdir: pathlib.Path,
                     dataset_name: str,
                     max_instances: int = 0,
                     num_jobs: int = 1,
                     schedulers: Optional[List[Scheduler]] = None,
                     overwrite: bool = False):
    """Evaluate a dataset.
    
    Args:
        datadir (pathlib.Path): The directory containing the dataset.
        resultsdir (pathlib.Path): The directory to save the results.
        dataset_name (str): The name of the dataset.
        max_instances (int, optional): Maximum number of instances to evaluate. Defaults to 0 (no trimming).
        num_jobs (int, optional): The number of jobs to run in parallel. Defaults to 1.
        schedulers (Optional[List[Scheduler]], optional): The schedulers to evaluate. Defaults to None (all schedulers).
        overwrite (bool, optional): Whether to overwrite existing results. Defaults to False.
    """
    logging.info("Evaluating dataset %s.", dataset_name)
    savepath = resultsdir.joinpath(f"{dataset_name}.csv")
    if savepath.exists() and not overwrite:
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
    savepath.parent.mkdir(exist_ok=True, parents=True)
    df_comp.to_csv(savepath)
    logging.info("Saved results to %s.", savepath)

def run_experiment(datadir: pathlib.Path,
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
    resultsdir.mkdir(parents=True, exist_ok=True)
    schedulers = schedulers if schedulers else get_schedulers()
    default_datasets = [path.stem for path in datadir.glob("*.json")]
    dataset_names = [dataset] if dataset else default_datasets
    for dataset_name in dataset_names:
        evaluate_dataset(
            datadir=datadir,
            resultsdir=resultsdir,
            dataset_name=dataset_name,
            max_instances=trim,
            num_jobs=num_jobs,
            schedulers=schedulers,
            overwrite=overwrite
        )

def main():
    logging.basicConfig(level=logging.INFO)

    datadir = thisdir.joinpath("data", "benchmarking")
    resultsdir = thisdir.joinpath("results", "benchmarking")
    outputdir = thisdir.joinpath("output", "benchmarking")

    prepare_datasets(savedir=datadir, skip_existing=True)
    run_experiment(datadir=datadir, resultsdir=resultsdir, num_jobs=1, overwrite=False)
    run_analysis(resultsdir=resultsdir, outputdir=outputdir)

if __name__ == "__main__":
    main()
