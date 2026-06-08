import logging
import pathlib
import random
import sys
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import filelock
import pandas as pd
from tqdm import tqdm

from saga import Scheduler
from saga.schedulers import HeftScheduler
from saga.schedulers.data import Dataset, ProblemInstance
from saga.schedulers.data.wfcommons import get_networks, get_workflows
from saga.schedulers.throughput.multi_obj import MultiObjScheduler
from saga.schedulers.throughput.inspirit import compute_inspiring_effeciency, compute_inspiring_ability

logging.basicConfig(level=logging.ERROR)

thisdir = pathlib.Path(__file__).parent.resolve()
datadir = thisdir / "data" / "throughput"
resultsdir = thisdir / "results" / "throughput"

WFCOMMONS_RECIPES = ["montage", "epigenomics", "cycles", "seismology"]

schedulers: Dict[str, Scheduler] = {
    "HEFT": HeftScheduler(),
    "MultiObj": MultiObjScheduler(),
}

_worker_resultsdir: pathlib.Path
_worker_schedulers: Dict[str, Scheduler]


def _init_worker(results_dir: pathlib.Path, sched: Dict[str, Scheduler]):
    global _worker_resultsdir, _worker_schedulers
    _worker_resultsdir = results_dir
    _worker_schedulers = sched


def prepare_wfcommons_dataset(
    recipe_name: str,
    num_instances: int = 10,
    ccr: float = 1.0,
    overwrite: bool = False,
) -> Dataset:
    dataset_name = f"wfcommons_{recipe_name}"
    dataset = Dataset(name=dataset_name)
    existing = set(dataset.instances) if not overwrite else set()
    instance_names = {f"{dataset_name}_{i}" for i in range(num_instances)}
    new_instances = instance_names - existing
    if not new_instances:
        return dataset

    networks = get_networks(num=len(new_instances), cloud_name="chameleon")
    workflows = get_workflows(num=len(new_instances), recipe_name=recipe_name)
    for i, instance_name in enumerate(new_instances):
        network = networks[i].scale_to_ccr(workflows[i], ccr)
        dataset.save_instance(
            ProblemInstance(name=instance_name, network=network, task_graph=workflows[i])
        )
    print(f"Prepared {dataset_name}: {dataset.size} instances.")
    return dataset


def _evaluate_instance(args: Tuple[str, str]) -> List[Dict]:
    dataset_name, instance_name = args
    dataset = Dataset(name=dataset_name)
    instance = dataset.get_instance(instance_name)
    savepath = _worker_resultsdir / f"{dataset_name}.csv"
    lock_path = savepath.with_suffix(".csv.lock")

    results = []
    for scheduler_name, scheduler in _worker_schedulers.items():
        with filelock.FileLock(lock_path):
            if savepath.exists():
                finished_df = pd.read_csv(savepath)
                finished = set(zip(finished_df["Dataset"], finished_df["Instance"], finished_df["Scheduler"]))
                if (dataset_name, instance_name, scheduler_name) in finished:
                    continue

        try:
            schedule = scheduler.schedule(network=instance.network, task_graph=instance.task_graph)
            makespan = schedule.makespan
            throughput = schedule.throughput
        except Exception as e:
            logging.warning("Failed %s/%s/%s: %s", dataset_name, instance_name, scheduler_name, e)
            continue

        result = {
            "Dataset": dataset_name,
            "Instance": instance_name,
            "Scheduler": scheduler_name,
            "Makespan": makespan,
            "Throughput": throughput,
        }
        results.append(result)

        with filelock.FileLock(lock_path):
            result_df = pd.DataFrame([result])
            if savepath.exists():
                result_df.to_csv(savepath, mode="a", header=False, index=False)
            else:
                result_df.to_csv(savepath, index=False)

    return results


def evaluate_dataset(dataset_name: str, num_workers: int = 4, seed: int = 42):
    savepath = resultsdir / f"{dataset_name}.csv"
    savepath.parent.mkdir(exist_ok=True, parents=True)

    dataset = Dataset(name=dataset_name)
    work_items = [(dataset_name, name) for name in dataset.instances]
    random.Random(seed).shuffle(work_items)

    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(resultsdir, schedulers),
    ) as pool:
        list(tqdm(
            pool.imap_unordered(_evaluate_instance, work_items),
            total=len(work_items),
            desc=f"Evaluating {dataset_name}",
            unit="instance",
        ))


def print_summary():
    print("\n=== Results Summary ===")
    for recipe in WFCOMMONS_RECIPES:
        dataset_name = f"wfcommons_{recipe}"
        savepath = resultsdir / f"{dataset_name}.csv"
        if not savepath.exists():
            continue
        df = pd.read_csv(savepath)
        cols = {"Makespan": "mean", "Throughput": "mean"} if "Throughput" in df.columns else {"Makespan": "mean"}
        pivot = df.groupby("Scheduler").agg(cols).reset_index()
        print(f"\n{dataset_name}:")
        print(pivot.to_string(index=False))


def main():
    import os
    os.environ["SAGA_DATA_DIR"] = str(datadir)
    datadir.mkdir(exist_ok=True, parents=True)

    print("Preparing wfcommons datasets...")
    for recipe in WFCOMMONS_RECIPES:
        try:
            prepare_wfcommons_dataset(recipe, num_instances=10, ccr=1.0)
        except Exception as e:
            print(f"Failed to prepare {recipe}: {e}")

    print("\nRunning benchmarks...")
    for recipe in WFCOMMONS_RECIPES:
        dataset_name = f"wfcommons_{recipe}"
        try:
            evaluate_dataset(dataset_name)
        except Exception as e:
            print(f"Failed to evaluate {recipe}: {e}")

    print_summary()


if __name__ == "__main__":
    main()
