import logging
import pathlib
import random
import zlib
from multiprocessing import Pool
from typing import Dict, List, Optional, Set, Tuple

import filelock
import pandas as pd
from common import (
    datadir,
    exclude_datasets,
    num_processors,
    num_samples,
    resultsdir,
    saga_schedulers,
    SchedulerFactory,
)
from tqdm import tqdm

from saga.schedulers.data import Dataset

# Global variables for worker processes
_worker_resultsdir: pathlib.Path
_worker_schedulers: Dict[str, SchedulerFactory]


def _init_worker(
    results_dir: pathlib.Path, schedulers: Dict[str, SchedulerFactory]
):
    """Initialize worker process with shared state."""
    global _worker_resultsdir, _worker_schedulers
    _worker_resultsdir = results_dir
    _worker_schedulers = schedulers


def _sample_seed(dataset_name: str, instance_name: str, sample: int) -> int:
    """Stable seed for one sample of one instance, independent of PYTHONHASHSEED."""
    return zlib.crc32(f"{dataset_name}/{instance_name}/{sample}".encode())


def _evaluate_instance(args: Tuple[str, str]) -> List[Dict]:
    """Evaluate a single instance with all schedulers.

    Randomized schedulers are run ``num_samples[name]`` times with distinct, reproducible
    seeds; each sample is recorded as its own row with its sample index and seed.

    Args:
        args: Tuple of (dataset_name, instance_name)

    Returns:
        List of result dictionaries.
    """
    dataset_name, instance_name = args
    dataset = Dataset(name=dataset_name)
    instance = dataset.get_instance(instance_name)
    savepath = _worker_resultsdir.joinpath(f"{dataset_name}.csv")
    lock_path = savepath.with_suffix(".csv.lock")

    # Read the finished set once per instance (with lock to avoid a race with writers).
    finished: Set[Tuple[str, str, str, int]] = set()
    with filelock.FileLock(lock_path):
        if savepath.exists():
            finished_df = pd.read_csv(savepath)
            finished = set(
                zip(
                    finished_df["Dataset"],
                    finished_df["Instance"],
                    finished_df["Scheduler"],
                    finished_df["Sample"],
                )
            )

    results = []
    for scheduler_name, factory in _worker_schedulers.items():
        for sample in range(num_samples.get(scheduler_name, 1)):
            if (dataset_name, instance_name, scheduler_name, sample) in finished:
                logging.info(
                    "  %s/%s/%s/%d already finished. Skipping.",
                    dataset_name,
                    instance_name,
                    scheduler_name,
                    sample,
                )
                continue
            seed = _sample_seed(dataset_name, instance_name, sample)
            scheduler = factory(seed)
            schedule = scheduler.schedule(
                network=instance.network, task_graph=instance.task_graph
            )
            result = {
                "Dataset": dataset_name,
                "Instance": instance_name,
                "Scheduler": scheduler_name,
                "Sample": sample,
                "Seed": seed,
                "Makespan": schedule.makespan,
            }
            results.append(result)

            # Write result immediately with lock
            with filelock.FileLock(lock_path):
                result_df = pd.DataFrame([result])
                if savepath.exists():
                    result_df.to_csv(savepath, mode="a", header=False, index=False)
                else:
                    result_df.to_csv(savepath, index=False)

            logging.info(
                "  %s/%s/%s/%d: %.4f",
                dataset_name,
                instance_name,
                scheduler_name,
                sample,
                schedule.makespan,
            )

    return results


def evaluate_dataset(
    resultsdir: pathlib.Path,
    dataset_name: str,
    schedulers: Optional[Dict[str, SchedulerFactory]] = None,
    overwrite: bool = False,
    num_workers: int = num_processors,
    shuffle: bool = True,
    seed: int = 42,
):
    """Evaluate a dataset using multiprocessing.

    Args:
        resultsdir: The directory to save the results.
        dataset_name: The name of the dataset.
        schedulers: Scheduler factories (seed -> Scheduler) to evaluate. Defaults to
            None (all schedulers in common.py).
        overwrite: Whether to overwrite existing results. Defaults to False.
        num_workers: Number of worker processes.
        shuffle: Whether to shuffle instances to distribute work evenly.
        seed: Random seed for shuffling.
    """
    savepath = resultsdir.joinpath(f"{dataset_name}.csv")
    savepath.parent.mkdir(exist_ok=True, parents=True)

    if overwrite and savepath.exists():
        savepath.unlink()

    schedulers = schedulers or saga_schedulers
    dataset = Dataset(name=dataset_name)
    instance_names = dataset.instances

    logging.info("Dataset %s has %d instances.", dataset_name, len(instance_names))

    # Create list of (dataset_name, instance_name) tuples
    work_items = [(dataset_name, name) for name in instance_names]

    # Shuffle to distribute expensive instances across workers
    if shuffle:
        random.Random(seed).shuffle(work_items)

    # Run in parallel with progress bar
    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(resultsdir, schedulers),
    ) as pool:
        list(
            tqdm(
                pool.imap_unordered(_evaluate_instance, work_items),
                total=len(work_items),
                desc=f"Evaluating {dataset_name}",
                unit="instance",
            )
        )


def main():
    datasets = [path.name for path in datadir.iterdir() if path.is_dir()]
    for dataset in datasets:
        if dataset in exclude_datasets:
            logging.info("Skipping excluded dataset %s", dataset)
            continue
        logging.info("Evaluating dataset %s", dataset)
        evaluate_dataset(resultsdir, dataset, overwrite=False)


if __name__ == "__main__":
    main()
