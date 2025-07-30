"""
Online vs Offline Scheduling Experiments

This script runs systematic experiments comparing online and offline scheduling algorithms.
It tests how well online schedulers perform compared to offline schedulers when task
execution times are uncertain (estimated vs actual).
"""

import os
import csv
import pathlib
import multiprocessing as mp
from itertools import product
from typing import Callable, Dict, List, Tuple
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
from saga.utils.random_variable import RandomVariable

# ---------------------- Config ----------------------
THISDIR = pathlib.Path(__file__).resolve().parent
CSV_PATH = THISDIR / "results.csv"
WORKFLOWS = list(recipes.keys()) # Available workflows from the WfCommons dataset (Montage, CyberShake, etc.)
CCRS = [0.2, 0.5, 1.0, 2.0, 5.0] # Low CCR = computation intensive, High CCR = communication intensive
N_SAMPLES = 100
ESTIMATE_METHODS: Dict[str, Callable[[RandomVariable], float]] = { #Either use mean, or mean + std deviation to estimate offline scheduling
    "mean": lambda x: x.mean(),
    "SHEFT": lambda x: x.mean() + x.std() if x.var()/x.mean() <= 1 else x.mean() * (1 + 1/x.std())
}
# ----------------------------------------------------

# Shared progress state
progress_counter = Value("i", 0)
progress_lock = Lock()
def print_progress(total_jobs: int):
    """ Print the progress of the experiments.

    Args:
        total_jobs (int): The total number of jobs to complete.
    """
    with progress_lock:
        progress_counter.value += 1
        current = progress_counter.value
        pct = (current / total_jobs) * 100
        print(f"Progress: {current}/{total_jobs} ({pct:.1f}%)", end="\r")


def init_csv(lock: mp.Lock):
    """ Initialize the CSV file for storing results.
    This function checks if the CSV file exists, and if not, creates it with the appropriate headers.
    
    Args:
        lock (mp.Lock): A multiprocessing lock to ensure thread-safe writing to the CSV file.
    """
    with lock:
        if not CSV_PATH.exists():
            with open(CSV_PATH, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "workflow", "estimate_method", "ccr", "sample", "scheduler_variant", "scheduler_type", "makespan"
                ])


def get_scheduler_variants() -> List[Tuple[str, Scheduler, Scheduler]]:
    """
    Returns a list of (variant_name, offline_scheduler, online_scheduler)

    Returns:
        List[Tuple[str, Scheduler, Scheduler]]: A list of tuples where each tuple contains
            the name of the scheduler variant, an instance of the offline scheduler,
            and an instance of the online scheduler.
    """
    variants = []

    # HEFT
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

    # CPoP
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


def run_one_experiment(workflow: str,
                       estimate_method_name: str,
                       ccr: float,
                       sample_index: int,
                       lock: mp.Lock) -> None:
    """
    Run a single experiment for the given workflow, CCR, and sample index.
    This function will schedule the tasks using both offline and online schedulers,
    and write the results to a CSV file.

    This is the core function that:
    1. Creates a problem instance (network + task graph)
    2. Runs offline, online, and naive online schedulers
    3. Measures makespan (total execution time) for each
    4. Writes results to CSV
    
    Args:
        workflow (str): The name of the workflow to run.
        estimate_method_name (str): The name of the method to estimate task weights.
        ccr (float): The communication-to-computation ratio for the workflow.
        sample_index (int): The index of the sample to run.
        lock (mp.Lock): A multiprocessing lock to ensure thread-safe writing to the CSV file.
    """
    try:
        # Get scheduler variants (HEFT and CPoP)
        scheduler_variants = get_scheduler_variants()

        # Get the estimation function (converts random variables to estimates)
        estimate_method = ESTIMATE_METHODS[estimate_method_name]

        # Create problem instance: network + task graph with estimated weights
        network, task_graph = get_wfcommons_instance(recipe_name=workflow, ccr=ccr, estimate_method=estimate_method)
        results = []

        for variant_name, offline_sched, online_sched in scheduler_variants:
            # Offline (uses actual weights)
            net_off, tg_off = get_offline_instance(network, task_graph)
            sched_offline = offline_sched.schedule(net_off, tg_off)
            ms_offline = max(task.end for tasks in sched_offline.values() for task in tasks)
            results.append((workflow, estimate_method_name, ccr, sample_index, variant_name, "Offline", ms_offline))

            # Online (similar to Naive Online except it reschedules every time a task finishes)
            sched_online = online_sched.schedule(network, task_graph)
            sched_online_actual = schedule_estimate_to_actual(network, task_graph, sched_online)
            ms_online = max(task.end for tasks in sched_online_actual.values() for task in tasks)
            results.append((workflow, estimate_method_name, ccr, sample_index, variant_name, "Online", ms_online))

            # Naive Online (use estimated weights than convert to actual schedule)
            sched_naive = offline_sched.schedule(network, task_graph)
            sched_naive_actual = schedule_estimate_to_actual(network, task_graph, sched_naive)
            ms_naive = max(task.end for tasks in sched_naive_actual.values() for task in tasks)
            results.append((workflow, estimate_method_name, ccr, sample_index, variant_name, "Naive Online", ms_naive))

        # Write to CSV
        with lock:
            with open(CSV_PATH, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(results)

    except Exception as e:
        print(f"Error in ({workflow}, {ccr}, {sample_index}): {e}")


def wrapped(args: Tuple[str, str, float, int, mp.Lock, int]) -> None:
    """
    Wrapper function to run an experiment in a multiprocessing context.
    This function unpacks the arguments and calls `run_one_experiment`.

    Args:
        args (Tuple[str, float, int, mp.Lock, int]): A tuple containing the
            workflow name, CCR, sample index, lock, and total number of jobs.
    """
    workflow, estimate_method_name, ccr, sample_index, lock, total_jobs = args
    run_one_experiment(workflow, estimate_method_name, ccr, sample_index, lock)
    print_progress(total_jobs)


def run_all_experiments():
    """
    Main function that orchestrates all experiments.
    
    This function:
    1. Creates all parameter combinations (workflows * estimation methods * CCRs * samples)
    2. Sets up multiprocessing for parallel execution
    3. Runs all experiments in parallel
    4. Saves results to CSV for analysis
    """
    # Create all parameter combinations
    # This generates: (workflow, estimate_method, ccr, sample_index) for each experiment
    all_params = list(product(WORKFLOWS, ESTIMATE_METHODS.keys(), CCRS, range(N_SAMPLES)))
    total_jobs = len(all_params)
    print(f"Total experiments: {total_jobs}")

    # Set up multiprocessing infrastructure
    manager = mp.Manager()
    lock = manager.Lock()
    init_csv(lock)

    # Prepare arguments for each experiment
    wrapped_args = [(w, em, c, s, lock, total_jobs) for w, em, c, s in all_params]

    # Run experiments in parallel (using 80% of CPU cores)
    with mp.Pool(processes=int(os.cpu_count() * 0.8)) as pool:
        pool.map(wrapped, wrapped_args)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    run_all_experiments()
