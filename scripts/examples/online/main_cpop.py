from functools import partial
from typing import Dict, Hashable, List, Tuple
from copy import deepcopy
from itertools import product
import os
import time
import pathlib
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from saga.utils.draw import gradient_heatmap
from multiprocessing import Pool, Value, Lock, cpu_count
from saga.schedulers import CpopScheduler, OnlineCpopScheduler
from saga.scheduler import Scheduler, Task
from saga.utils.random_graphs import get_branching_dag, get_network
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from saga.utils.online_tools import schedule_estimate_to_actual, get_offline_instance

from saga.schedulers.data.wfcommons import get_workflows, get_networks, get_wfcommons_instance
from saga.utils.random_variable import RandomVariable

thisdir = pathlib.Path(__file__).resolve().parent



def safe_value(x: float) -> float:
    """
    Helper function for get_instance() that hardcodes any value less than or equal to zero to 1e-9
    """
    return x if x > 0 else 1e-9
 
    
def get_instance(levels: int, branching_factor: int, ccr: float = 1.0) -> Tuple[nx.Graph, nx.DiGraph]:
    """
    Create a network and a branching task graph with the specified number
    of levels and branching factor.
    """
    network = get_network(num_nodes=4)
    task_graph = get_branching_dag(levels=levels, branching_factor=branching_factor)

    
    #Make graph for varying these
    node_loc = RandomVariable(samples=np.random.normal(size=100000, loc=24, scale=5))
    node_scale = RandomVariable(samples=np.random.normal(size=100000, loc=2, scale=0.2))
    #Do failsafe check to make sure weights are not negative possibly use 1e-9

    for node in network.nodes:
        random_node = RandomVariable(samples=np.random.normal(size=100000, loc=node_loc.sample(), scale=node_scale.sample()))
        network.nodes[node]["weight_estimate"] = safe_value(random_node.mean())
        network.nodes[node]["weight_actual"] = safe_value(random_node.sample())
        network.nodes[node]["weight"] = network.nodes[node]["weight_estimate"]
        

    for (u, v) in network.edges:
        if u == v:
            network.edges[u, v]["weight_estimate"] = 1e9
            network.edges[u, v]["weight_actual"] = 1e9
            network.edges[u, v]["weight"] = 1e9
        else:
            random_node = RandomVariable(samples=np.random.normal(size=100000, loc=node_loc.sample(), scale=node_scale.sample()))
            network.edges[u, v]["weight_estimate"] = safe_value(random_node.mean())
            network.edges[u, v]["weight_actual"] = safe_value(random_node.sample())
            network.edges[u, v]["weight"] = network.edges[u, v]["weight_estimate"]

    for task in task_graph.nodes:
        random_node = RandomVariable(samples=np.random.normal(size=100000, loc=node_loc.sample() * ccr, scale=node_scale.sample())) # Do not understand what CCR is doing ???
        task_graph.nodes[task]["weight_estimate"] = safe_value(random_node.mean())
        task_graph.nodes[task]["weight_actual"] = safe_value(random_node.sample())
        task_graph.nodes[task]["weight"] = task_graph.nodes[task]["weight_estimate"]

    for (src, dst) in task_graph.edges:
        random_node = RandomVariable(samples=np.random.normal(size=100000, loc=node_loc.sample(), scale=node_scale.sample()))
        task_graph.edges[src, dst]["weight_estimate"] = safe_value(random_node.mean())
        task_graph.edges[src, dst]["weight_actual"] = safe_value(random_node.sample())
        task_graph.edges[src, dst]["weight"] = task_graph.edges[src, dst]["weight_estimate"]
    return network, task_graph

counter = Value('i', 0)  # Shared integer counter
total = Value('i', 0)  # Shared integer total
counter_lock = Lock()  # Lock to prevent race conditions
def progress_callback():
    with counter.get_lock():
        counter.value += 1
        perc = counter.value / total.value * 100
        print(f"Progress: {counter.value}/{total.value} ({perc:.2f}%)" + " " * 10, end="\r")

def run_sample(ccr: float,
               levels: int,
               branching_factor: int,
               sample_index: int) -> List[Tuple]:
    """
    Runs one experiment sample for a given set of parameters and updates the experiment counter.

    Args:
        params: A tuple (ccr, levels, branching_factor, sample_index, counter, counter_lock)
    
    Returns:
        A list of three tuples, one for each scheduler variant:
            (ccr, levels, branching_factor, sample_index, makespan, scheduler_name)
    """
    global counter, counter_lock

    scheduler = CpopScheduler()
    scheduler_online = OnlineCpopScheduler()
    network, task_graph = get_instance(levels, branching_factor, ccr=ccr)
    
    # Run standard HEFT (Naive Online HEFT)
    schedule_online_naive = scheduler.schedule(network, task_graph)
    schedule_online_naive_actual = schedule_estimate_to_actual(network, task_graph, schedule_online_naive)
    makespan_online_naive = max(task.end for node_tasks in schedule_online_naive_actual.values() for task in node_tasks)
    
    # Run Online HEFT
    schedule_online = scheduler_online.schedule(network, task_graph)
    makespan_online = max(task.end for node_tasks in schedule_online.values() for task in node_tasks)
    
    # Run Offline HEFT (knows actual weights)
    network_offline, task_graph_offline = get_offline_instance(network, task_graph)
    schedule_offline = scheduler.schedule(network_offline, task_graph_offline)
    makespan_offline = max(task.end for node_tasks in schedule_offline.values() for task in node_tasks)
    
    # Increment the counter in a thread-safe manner
    progress_callback()

    return [
        (ccr, levels, branching_factor, sample_index, makespan_online_naive, "Naive Online HEFT"),
        (ccr, levels, branching_factor, sample_index, makespan_online, "Online HEFT"),
        (ccr, levels, branching_factor, sample_index, makespan_offline, "Offline HEFT")
    ]

def run_experiment():
    cores = int(os.cpu_count() * 0.8)
    n_samples = 50
    experiments = []
    ccrs = [1/5, 1/2, 1, 2, 5]
    #levels_range = [1, 2, 3, 4]
    #branching_range = [1, 2, 3, 4]
    levels_range = [1, 2, 3, 4]
    branching_range = [1, 2]
    all_params = list(product(ccrs, levels_range, branching_range, range(n_samples)))
    print("TEST TEST TEST")
    print(f"Number of param combinations: {len(all_params)}")

    with counter.get_lock():
        counter.value = 0
        total.value = len(all_params)

    # Multiprocessing
    if cores == 1:
        results = [
            run_sample(*params) for params in all_params
            if run_sample(*params) is not None
        ]
    else:
        with Pool(processes=cores) as pool:
            results = pool.starmap(run_sample, all_params)

    # Flatten results
    for res in results:
        experiments.extend(res)

    # Save results to CSV
    df = pd.DataFrame(experiments, columns=["ccr", "levels", "branching_factor", "instance", "makespan", "scheduler"])
    df.to_csv(thisdir / "makespan_experiments_cpop.csv", index=False)

def analyze_results():
    df = pd.read_csv(thisdir / "makespan_experiments_cpop.csv")
    df["makespan_ratio"] = df.groupby(by=["ccr", "levels", "branching_factor", "instance"])["makespan"].transform(lambda x: x / x.min())

    ccrs = sorted(df["ccr"].unique())

    for ccr in ccrs:
        fig, (ax_naive, ax_online) = plt.subplots(1, 2, figsize=(20, 10))
        for scheduler, ax in [("Naive Online HEFT", ax_naive), ("Online HEFT", ax_online)]:
            df_scheduler = df[df["scheduler"] == scheduler]
            gradient_heatmap(
                df_scheduler,
                x="levels",
                y="branching_factor",
                color="makespan_ratio",
                title=f"{scheduler}",
                x_label="Levels", rotate_xlabels=0,
                y_label="Branching Factor",
                color_label="Makespan Ratio",
                xorder=lambda x: int(x),
                yorder=lambda x: -int(x),
                font_size=20,
                ax=ax,
                upper_threshold=10.0,
                # cmap_lower=1.0
            )
        
        fig.tight_layout()
        fig.savefig(thisdir / f"makespan_heatmap_ccr_{ccr}.png")

    # print stats on makespan ratio
    print("Makespan ratio stats:")
    print(df.groupby("scheduler")["makespan_ratio"].describe())

def run_example():
    levels = 4
    branching_factor = 4
    ccr = 1
    sample_index = 0
    
    scheduler = CpopScheduler()
    scheduler_online = OnlineCpopScheduler()

    #network, task_graph = get_instance(levels, branching_factor)
    network, task_graph = get_wfcommons_instance(recipe_name="montage", ccr=ccr)

    network_offline, task_graph_offline = get_offline_instance(network, task_graph)
    schedule_offline = scheduler.schedule(network_offline, task_graph_offline)
    makespan_offline = max(task.end for node_tasks in schedule_offline.values() for task in node_tasks)
    print(f"Offline HEFT makespan: {makespan_offline}")

    schedule_online_naive = scheduler.schedule(network, task_graph)
    schedule_online_naive_actual = schedule_estimate_to_actual(network, task_graph, schedule_online_naive)
    makespan_online_naive = max(task.end for node_tasks in schedule_online_naive_actual.values() for task in node_tasks)
    print(f"Naive Online HEFT makespan: {makespan_online_naive}")

    schedule_online = scheduler_online.schedule(network, task_graph)
    makespan_online = max(task.end for node_tasks in schedule_online.values() for task in node_tasks)
    print(f"Online HEFT makespan: {makespan_online}")



def main():
    # #record start time
    # start_time = time.perf_counter()

    run_example()
    #run_experiment()
    #analyze_results()

    # #record end time
    # end_time = time.perf_counter()
    # elapsed = end_time - start_time
    # print(f"Execution time: {elapsed:.2f} seconds")

    
    workflow = get_wfcommons_instance(recipe_name="montage", ccr=1)

if __name__ == "__main__":
    main()