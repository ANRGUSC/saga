import os
import csv
import pathlib
import multiprocessing as mp
from itertools import product
from typing import List, Tuple
from multiprocessing import Value, Lock

import pandas as pd
import numpy as np
from saga.scheduler import Scheduler, Task
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.online_parametric import OnlineParametricScheduler
from saga.schedulers.parametric.components import (
    CPoPRanking, UpwardRanking, GreedyInsert
)
from saga.utils.online_tools import schedule_estimate_to_actual, get_offline_instance
from saga.schedulers.data.wfcommons import get_wfcommons_instance, recipes

# ---------------------- Config ----------------------
THISDIR = pathlib.Path(__file__).resolve().parent
CSV_PATH = THISDIR / "results.csv"
# WORKFLOWS = ["montage", "blast", "epigenomics"]
WORKFLOWS = list(recipes.keys())
CCRS = [0.2, 0.5, 1.0, 2.0, 5.0]
N_SAMPLES = 50
# ----------------------------------------------------

# Shared progress state
progress_counter = Value("i", 0)
progress_lock = Lock()

def print_progress(total_jobs: int):
    with progress_lock:
        progress_counter.value += 1
        current = progress_counter.value
        pct = (current / total_jobs) * 100
        print(f"Progress: {current}/{total_jobs} ({pct:.1f}%)", end="\r")


def init_csv(lock: mp.Lock):
    with lock:
        if not CSV_PATH.exists():
            with open(CSV_PATH, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "workflow", "ccr", "sample", "scheduler_variant", "scheduler_type", "makespan"
                ])


def get_scheduler_variants() -> List[Tuple[str, Scheduler, Scheduler]]:
    """
    Returns a list of (variant_name, offline_scheduler, online_scheduler)
    """
    variants = []

    # HEFT-like (standard)
    variants.append((
        "HEFT",
        ParametricScheduler(
            initial_priority=UpwardRanking(),
            insert_task=GreedyInsert(append_only=False, compare="EFT", critical_path=False)
        ),
        OnlineParametricScheduler(
            initial_priority=UpwardRanking(),
            insert_task=GreedyInsert(append_only=False, compare="EFT", critical_path=False)
        )
    ))

    # CPoP-like
    variants.append((
        "CPoP",
        ParametricScheduler(
            initial_priority=CPoPRanking(),
            insert_task=GreedyInsert(append_only=False, compare="EFT", critical_path=True)
        ),
        OnlineParametricScheduler(
            initial_priority=CPoPRanking(),
            insert_task=GreedyInsert(append_only=False, compare="EFT", critical_path=True)
        )
    ))

    return variants


def run_one_experiment(
    workflow: str, ccr: float, sample_index: int, lock: mp.Lock
):
    try:
        scheduler_variants = get_scheduler_variants()
        network, task_graph = get_wfcommons_instance(recipe_name=workflow, ccr=ccr)
        results = []

        for variant_name, offline_sched, online_sched in scheduler_variants:
            # Offline
            net_off, tg_off = get_offline_instance(network, task_graph)
            sched_offline = offline_sched.schedule(net_off, tg_off)
            ms_offline = max(task.end for tasks in sched_offline.values() for task in tasks)
            results.append((workflow, ccr, sample_index, variant_name, "Offline", ms_offline))

            # Online
            sched_online = online_sched.schedule(network, task_graph)
            sched_online_actual = schedule_estimate_to_actual(network, task_graph, sched_online)
            ms_online = max(task.end for tasks in sched_online_actual.values() for task in tasks)
            results.append((workflow, ccr, sample_index, variant_name, "Online", ms_online))

            # Naive Online
            sched_naive = offline_sched.schedule(network, task_graph)
            sched_naive_actual = schedule_estimate_to_actual(network, task_graph, sched_naive)
            ms_naive = max(task.end for tasks in sched_naive_actual.values() for task in tasks)
            results.append((workflow, ccr, sample_index, variant_name, "Naive Online", ms_naive))

        # Write to CSV
        with lock:
            with open(CSV_PATH, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(results)

    except Exception as e:
        print(f"Error in ({workflow}, {ccr}, {sample_index}): {e}")


def wrapped(args):
    workflow, ccr, sample_index, lock, total_jobs = args
    run_one_experiment(workflow, ccr, sample_index, lock)
    print_progress(total_jobs)


def run_all_experiments():
    all_params = list(product(WORKFLOWS, CCRS, range(N_SAMPLES)))
    total_jobs = len(all_params)
    print(f"Total experiments: {total_jobs}")

    manager = mp.Manager()
    lock = manager.Lock()
    init_csv(lock)

    wrapped_args = [(w, c, s, lock, total_jobs) for w, c, s in all_params]

    with mp.Pool(processes=int(os.cpu_count() * 0.8)) as pool:
        pool.map(wrapped, wrapped_args)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    run_all_experiments()
