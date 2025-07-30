"""
Online vs Offline Scheduling Experiments

This script runs systematic experiments comparing online and offline scheduling algorithms.
It tests how well online schedulers perform compared to offline schedulers when task
execution times are uncertain (estimated vs actual).
"""
from dataclasses import dataclass
import os
import csv
import pathlib
import multiprocessing as mp
from itertools import product
from typing import Callable, Dict, Hashable, List, Tuple
from multiprocessing import Value, Lock
import networkx as nx

from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.online_parametric import OnlineParametricScheduler
from saga.schedulers.parametric.components import (
    GREEDY_INSERT_COMPARE_FUNCS, ArbitraryTopological, CPoPRanking, UpwardRanking, GreedyInsert
)
from saga.utils.online_tools import schedule_estimate_to_actual, get_offline_instance, set_weights
from saga.schedulers.data.wfcommons import get_wfcommons_instance, recipes
from saga.utils.random_variable import RandomVariable

# ---------------------- Config ----------------------
THISDIR = pathlib.Path(__file__).resolve().parent
CSV_PATH = THISDIR / "results.csv"
WORKFLOWS = list(recipes.keys()) # Available workflows from the WfCommons dataset (Montage, CyberShake, etc.)
CCRS = [0.2, 0.5, 1.0, 2.0, 5.0] # Low CCR = computation intensive, High CCR = communication intensive
N_SAMPLES = 100
ESTIMATE_METHODS: Dict[str, Callable[[RandomVariable, bool], float]] = {
    "mean": lambda x, is_cost: x.mean(),
    "SHEFT": lambda x, is_cost: x.mean() + (-1 if is_cost else 1) * x.std() if x.var()/x.mean() <= 1 else x.mean() * (1 + (-1 if is_cost else 1) * 1/x.std())
}
IGNORE_ERRORS = True  # Set to False to raise exceptions during experiments
RUN_RESTRICTED = True  # If True, only run HEFT and CPoP variants; otherwise run all combinations
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


def init_csv(lock: mp.Lock) -> bool:
    """ Initialize the CSV file for storing results.
    This function checks if the CSV file exists, and if not, creates it with the appropriate headers.
    
    Args:
        lock (mp.Lock): A multiprocessing lock to ensure thread-safe writing to the CSV file.

    Returns:
        bool: True if the CSV was initialized successfully, False if it already exists and user chose not to overwrite.
    """
    with lock:
        if CSV_PATH.exists():
            while True:
                res = input(f"CSV file {CSV_PATH} already exists. Overwrite? (y/n): ").strip().lower()
                if res == "y":
                    CSV_PATH.unlink()
                elif res == "n":
                    return False
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
        with CSV_PATH.open(mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "workflow", "estimate_method", "ccr", "sample", "scheduler", "scheduler_type",
                "ranking_function", "append_only", "compare", "critical_path",
                "makespan"
            ])

    return True


@dataclass
class SchedulerVariant:
    name: str
    ranking_function: Callable[[nx.Graph, nx.DiGraph], List[Hashable]]
    append_only: bool
    compare: str
    critical_path: bool

def get_scheduler_variants_restricted() -> List[SchedulerVariant]:
    """
    Returns a list of (variant_name, offline_scheduler, online_scheduler) tuples for restricted scheduler variants (HEFT and CPoP).

    Returns:
        List[SchedulerVariant]: A list of restricted scheduler variants.
            the name of the scheduler variant, an instance of the offline scheduler,
            and an instance of the online scheduler.
    """
    variants: List[SchedulerVariant] = []

    # HEFT
    variants.append(SchedulerVariant(
        name="HEFT",
        ranking_function=UpwardRanking(),
        append_only=False,
        compare="EFT",
        critical_path=False
    ))

    # CPoP
    variants.append(SchedulerVariant(
        name="CPoP",
        ranking_function=CPoPRanking(),
        append_only=False,
        compare="EFT",
        critical_path=True
    ))

    return variants

def get_scheduler_variants_all() -> List[SchedulerVariant]:
    """
    Returns a list of all scheduler variants, including both restricted and full sets.

    Returns:
        List[SchedulerVariant]: A list of all scheduler variants.
            the name of the scheduler variant, an instance of the offline scheduler,
            and an instance of the online scheduler.
    """
    ranking_funcs = [
        UpwardRanking(),
        CPoPRanking(),
        ArbitraryTopological()
    ]
    append_only_options = [False, True]
    compare_options = GREEDY_INSERT_COMPARE_FUNCS
    critical_path_options = [False, True]
    variants: List[SchedulerVariant] = []
    for ranking, append_only, compare, critical_path in product(
        ranking_funcs,
        append_only_options,
        compare_options,
        critical_path_options
    ):
        variants.append(
            SchedulerVariant(
                name=f"{ranking.__class__.__name__}_{append_only}_{compare}_{critical_path}",
                ranking_function=ranking,
                append_only=append_only,
                compare=compare,
                critical_path=critical_path
            )
        )

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
        scheduler_variants = get_scheduler_variants_restricted() if RUN_RESTRICTED else get_scheduler_variants_all()
        estimate_method = ESTIMATE_METHODS[estimate_method_name]

        # Create problem instance: network + task graph with estimated weights
        network, task_graph = get_wfcommons_instance(recipe_name=workflow, ccr=ccr, estimate_method=estimate_method)
        results = []

        for variant in scheduler_variants:
            offline_scheduler = ParametricScheduler(
                initial_priority=variant.ranking_function,
                insert_task=GreedyInsert(
                    append_only=variant.append_only,
                    compare=variant.compare,
                    critical_path=variant.critical_path
                )
            )
            online_scheduler = OnlineParametricScheduler(
                initial_priority=variant.ranking_function,
                insert_task=GreedyInsert(
                    append_only=variant.append_only,
                    compare=variant.compare,
                    critical_path=variant.critical_path
                )
            )

            net_online = set_weights(network, "weight_estimate")
            tg_online = set_weights(task_graph, "weight_estimate")
            net_offline = set_weights(network, "weight_actual")
            tg_offline = set_weights(task_graph, "weight_actual")

            # Offline
            sched_offline = offline_scheduler.schedule(network=net_offline, task_graph=tg_offline)
            ms_offline = max(task.end for tasks in sched_offline.values() for task in tasks)
            results.append(
                (
                    workflow, estimate_method_name, ccr, sample_index, variant.name, "Offline",
                    variant.ranking_function.__class__.__name__, variant.append_only, variant.compare,
                    variant.critical_path, ms_offline
                )
            )

            # Online
            sched_online = online_scheduler.schedule(network=net_online, task_graph=tg_online)
            ms_online = max(task.end for tasks in sched_online.values() for task in tasks)
            results.append(
                (
                    workflow, estimate_method_name, ccr, sample_index, variant.name, "Online",
                    variant.ranking_function.__class__.__name__, variant.append_only, variant.compare,
                    variant.critical_path, ms_online
                )
            )

            # Naive Online
            sched_naive = offline_scheduler.schedule(network=net_online, task_graph=tg_online)
            sched_naive_actual = schedule_estimate_to_actual(network, task_graph, sched_naive)
            ms_naive = max(task.end for tasks in sched_naive_actual.values() for task in tasks)
            results.append(
                (
                    workflow, estimate_method_name, ccr, sample_index, variant.name, "Naive Online",
                    variant.ranking_function.__class__.__name__, variant.append_only, variant.compare,
                    variant.critical_path, ms_naive
                )
            )

        # Write to CSV
        with lock:
            with open(CSV_PATH, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(results)

    except Exception as e:
        print(f"Error in ({workflow}, {ccr}, {sample_index}): {e}")
        if not IGNORE_ERRORS:
            raise e


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


def run_restricted_experiments():
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

    if not init_csv(lock):
        return

    # Prepare arguments for each experiment
    wrapped_args = [(w, em, c, s, lock, total_jobs) for w, em, c, s in all_params]

    # Run experiments in parallel (using 80% of CPU cores)
    with mp.Pool(processes=int(os.cpu_count() * 0.8)) as pool:
        pool.map(wrapped, wrapped_args)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    run_restricted_experiments()
