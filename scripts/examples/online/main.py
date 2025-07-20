from functools import partial
from typing import Dict, Hashable, List, Tuple
from copy import deepcopy
from itertools import product
import os
import time
import pathlib
import networkx as nx
import matplotlib.pyplot as plt
from statistics import mean
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd
from saga.utils.draw import gradient_heatmap, draw_gantt, draw_network, draw_task_graph
from multiprocessing import Pool, Value, Lock, cpu_count
from saga.schedulers import online_heft
from saga.schedulers import HeftScheduler, online_heft, OnlineHeftScheduler, CpopScheduler, OnlineCpopScheduler, ETFScheduler, OnlineETFScheduler, SufferageScheduler, OnlineSufferageScheduler
from saga.schedulers import TempHeftScheduler, OnlineTempHeftScheduler
from saga.scheduler import Scheduler, Task
from saga.utils.random_graphs import get_branching_dag, get_network
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from saga.utils.online_tools import schedule_estimate_to_actual, get_offline_instance

from saga.schedulers.data.wfcommons import get_workflows, get_networks, get_wfcommons_instance
from saga.utils.random_variable import RandomVariable
from saga.schedulers.parametric.online_parametric import OnlineParametricScheduler
from saga.schedulers.parametric.components import ArbitraryTopological, GreedyInsert, CPoPRanking, UpwardRanking, ArbitraryTopological
from saga.schedulers.parametric import ParametricScheduler

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
        random_node = RandomVariable(samples=np.random.normal(size=100000, loc=node_loc.sample() * ccr, scale=node_scale.sample())) 
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

    #scheduler = HeftScheduler()
    #scheduler_online = OnlineHeftScheduler()
    #scheduler = CpopScheduler()
    #scheduler_online = OnlineCpopScheduler()
    #scheduler = ETFScheduler()
    #scheduler_online = OnlineETFScheduler()
    #scheduler = SufferageScheduler()
    #scheduler_online = OnlineSufferageScheduler()
    #scheduler = TempHeftScheduler()
    #scheduler_online = OnlineTempHeftScheduler()
    scheduler = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(
            append_only=False,
            compare="EFT",
            critical_path=False
        )
    )
    scheduler_online = OnlineParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(
            append_only=False,
            compare="EFT",
            critical_path=False
        )
    )
    
    network, task_graph = get_wfcommons_instance(recipe_name="montage", ccr=ccr)
    #network, task_graph = get_instance(levels, branching_factor, ccr=ccr)
    
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
    #ccrs = [1/5, 1/2, 1, 2, 5]
    ccrs = [1/5]
    #levels_range = [1, 2, 3, 4]
    #branching_range = [1, 2, 3, 4]
    levels_range = [1, 2, 3, 4]
    branching_range = [1, 2, 3,4]
    all_params = list(product(ccrs, levels_range, branching_range, range(n_samples)))
    #print("TEST TEST TEST")
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
    df.to_csv(thisdir / "makespan_experiments.csv", index=False)

def analyze_results():
    #df = pd.read_csv(thisdir / "makespan_experiments.csv")
    #df["makespan_ratio"] = df.groupby(by=["ccr", "levels", "branching_factor", "instance"])["makespan"].transform(lambda x: x / x.min())
    df = pd.read_csv(thisdir / "makespan_experiments.csv")

    # 1) Extract the offline baseline makespan for each (ccr, levels, branching_factor, instance)
    offline_baseline = (
        df[df["scheduler"] == "Offline HEFT"]
          .set_index(["ccr", "levels", "branching_factor", "instance"])["makespan"]
          .rename("offline_makespan")
    )

    # 2) Merge the offline baseline back onto the full DataFrame
    df = df.merge(
        offline_baseline,
        how="left",
        left_on=["ccr", "levels", "branching_factor", "instance"],
        right_index=True
    )

    # 3) Compute makespan_ratio = (this run’s makespan) / (offline makespan)
    df["makespan_ratio"] = df["makespan"] / df["offline_makespan"]

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


    #results with makespan ratios of 1 removed 

    df_nonbaseline = df[df["makespan_ratio"] != 1.0]

    # (C) Re‐plot or re‐summarize using only the “non‐baseline” rows:
    for ccr in ccrs:
        fig, (ax_naive, ax_online) = plt.subplots(1, 2, figsize=(20, 10))
        for scheduler, ax in [("Naive Online HEFT", ax_naive), ("Online HEFT", ax_online)]:
            # take only rows with ratio > 1.0 (i.e., scheduler ran strictly slower than offline)
            df_sched_nonbaseline = df_nonbaseline[
                (df_nonbaseline["scheduler"] == scheduler) & 
                (df_nonbaseline["ccr"] == ccr)
            ]
            if df_sched_nonbaseline.empty:
                # if there are no non‐baseline rows for this combo, you can skip or plot an empty heatmap
                ax.set_title(f"{scheduler} (no worse‐than‐baseline cases)")
                continue

            gradient_heatmap(
                df_sched_nonbaseline,
                x="levels",
                y="branching_factor",
                color="makespan_ratio",
                title=f"{scheduler} (CCR={ccr}, ratio>1)",
                x_label="Levels", rotate_xlabels=0,
                y_label="Branching Factor",
                color_label="Makespan Ratio",
                xorder=lambda x: int(x),
                yorder=lambda x: -int(x),
                font_size=20,
                ax=ax,
                upper_threshold=10.0,
            )
        
        fig.tight_layout()
        fig.savefig(thisdir / f"makespan_heatmap_nonbaseline_ccr_{ccr}.png")

    # (D) Print stats restricted to ratio > 1
    print("\nMakespan ratio stats (only ratio > 1):")
    print(df_nonbaseline.groupby("scheduler")["makespan_ratio"].describe())



def run_example():
    levels = 4
    branching_factor = 2
    ccr = 1
    sample_index = 0
    
    #scheduler = HeftScheduler()
    #scheduler_online = OnlineHeftScheduler()
    #scheduler = CpopScheduler()
    #scheduler_online = OnlineCpopScheduler()
    # scheduler = ETFScheduler()
    # scheduler_online = OnlineETFScheduler()
    #scheduler = SufferageScheduler()
    #scheduler_online = OnlineSufferageScheduler()
    scheduler = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(
            append_only=False,
            compare="EFT",
            critical_path=False
        )
    )
    # scheduler_online = OnlineTempHeftScheduler()
    scheduler_online = OnlineParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(
            append_only=False,
            compare="EFT",
            critical_path=False
        )
    )

    network, task_graph = get_instance(levels, branching_factor)
    #network, task_graph = get_wfcommons_instance(recipe_name="montage", ccr=ccr)
    
    network_offline, task_graph_offline = get_offline_instance(network, task_graph)
    schedule_offline = scheduler.schedule(network_offline, task_graph_offline)
    #print(schedule_offline)
    makespan_offline = max(task.end for node_tasks in schedule_offline.values() for task in node_tasks)
    print(f"Offline makespan: {makespan_offline}")

    schedule_online_naive = scheduler.schedule(network, task_graph)
    schedule_online_naive_actual = schedule_estimate_to_actual(network, task_graph, schedule_online_naive)
    #print(schedule_online_naive_actual)
    makespan_online_naive = max(task.end for node_tasks in schedule_online_naive_actual.values() for task in node_tasks)
    print(f"Naive Online makespan: {makespan_online_naive}")

    schedule_online = scheduler_online.schedule(network, task_graph)
    makespan_online = max(task.end for node_tasks in schedule_online.values() for task in node_tasks)
    #print(schedule_online)
    print(f"Online makespan: {makespan_online}")
     #1. Prepare the actual‐time schedules
    sched_off    = schedule_offline
    sched_naive  = schedule_estimate_to_actual(network, task_graph, schedule_online_naive)
    sched_online = schedule_estimate_to_actual(network, task_graph, schedule_online)

   
    # 3. Helper to compute makespan
    def makespan(schedule: Dict[Hashable, List[Task]]) -> float:
        return max(
            task.end
            for tasks in schedule.values()
            for task in tasks
        )

    # 4. Find the global max end‐time across all three
    all_schedules = [sched_off, sched_naive, sched_online]
    global_max    = max(makespan(s) for s in all_schedules)

    # 5. Draw each Gantt with identical x‐limits
    for filename, sched in [
        ("schedule_offline.png",      sched_off),
        ("schedule_online_naive.png", sched_naive),
        ("schedule_online.png",       sched_online),
    ]:
        ax = draw_gantt(schedule=sched)
        ax.set_xlim(0, global_max)
        ax.get_figure().savefig(thisdir / filename)
        
    # ax = draw_gantt(
    #     schedule=schedule_online
    # )
    # fig = ax.get_figure()
    # fig.savefig(thisdir / "schedule_online.png")
    # ax = draw_gantt(
    #     schedule=schedule_online_naive
    # )
    # fig = ax.get_figure()
    # fig.savefig(thisdir / "schedule_online_naive.png")
    # ax = draw_gantt(
    #     schedule=schedule_offline
    # )
    # fig = ax.get_figure()
    # fig.savefig(thisdir / "schedule_offline.png")
        
def run_fixed_suite(scheduler: Scheduler, n_samples: int = 50) -> List[float]:
    """
    Runs n_samples of the 4-level, branching-4 DAG on the given scheduler,
    returning the list of makespans.
    """
    makespans = []
    for i in range(n_samples):
        network, task_graph = get_instance(levels=4, branching_factor=4, ccr=1.0)
        # schedule & turn estimates into actual‐time
        sched_est = scheduler.schedule(network, task_graph)
        sched_act = schedule_estimate_to_actual(network, task_graph, sched_est)
        # compute makespan
        mp = max(t.end for tasks in sched_act.values() for t in tasks)
        makespans.append(mp)
    return makespans

def evaluate_parametric_schedulers(n_samples: int = 50):
    rows = []

    priorities = [("Upward",   UpwardRanking),
                  ("Arbitrary Topological", ArbitraryTopological),
                  ("Cpop",     CPoPRanking)]
    compares   = ["EFT", "EST", "Quickest"]
    combos = list(product(priorities, [False, True], compares, [False, True]))
    total = len(combos)

    for idx, ((pname, Pclass), append_only, compare, critical) in enumerate(combos, start=1):
        # progress print
        print(
            f"Combo {idx}/{total}: {pname}, append_only={append_only}, "
            f"compare={compare}, critical_path={critical}", end="\r"
                )

        # build scheduler
        sched = ParametricScheduler(
            initial_priority=Pclass(),
            insert_task=GreedyInsert(
                append_only=append_only,
                compare=compare,
                critical_path=critical
            )
        )

        # run and record
        mps = run_fixed_suite(sched, n_samples=n_samples)
        rows.append({
            "priority":      pname,
            "append_only":   append_only,
            "compare":       compare,
            "critical_path": critical,
            "avg_makespan":  mean(mps)
        })

    print("\nDone.\n")  # move to new line when finished
    df = pd.DataFrame(rows)
    df.to_csv(thisdir / "parametric_makespan_comparison.csv", index=False)
    print("Wrote results to parametric_makespan_comparison.csv")

def plot_performance_histogram(
    csv_path: Path,
    output_path: Path
):
    """
    Reads the CSV of average makespans, creates full descriptive labels, selects the top 3,
    bottom 3, and 3 middle configurations, and plots a vertical bar chart with bars colored
    by performance (blue=fastest, red=slowest), saving as a PNG.
    """
    df = pd.read_csv(csv_path)

    def make_label(row):
        priority = {"Upward": "Upward", "Downward": "Downward", "Custom": "Custom"}.get(row["priority"], row["priority"])
        append = "Append-Only" if row["append_only"] else "Insert"
        compare = {
            "EFT": "Earliest Finish Time",
            "EST": "Earliest Start Time",
            "Quickest": "Quickest Path"
        }.get(row["compare"], row["compare"])
        critical = "Critical Path First" if row["critical_path"] else "No Critical Path"
        return f"{priority}, {append}, {compare}, {critical}"

    df["label"] = df.apply(make_label, axis=1)
    df_sorted = df.sort_values("avg_makespan").reset_index(drop=True)

    total = len(df_sorted)
    top3 = df_sorted.iloc[:3]
    bottom3 = df_sorted.iloc[-3:]
    mid_start = total // 2 - 1
    mid3 = df_sorted.iloc[mid_start:mid_start + 3]

    selected = pd.concat([top3, mid3, bottom3])
    selected = selected.reset_index(drop=True)

    labels = selected["label"].tolist()
    makespans = selected["avg_makespan"].values

    norm = plt.Normalize(vmin=makespans.min(), vmax=makespans.max())
    cmap = plt.cm.get_cmap("coolwarm")
    colors = cmap(norm(makespans))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(labels)), makespans, color=colors)
    ax.set_title("Online Parametric Performance (4x4 Task Graph, CCR = 1.0, Best, Median, and Worst Results)", fontsize=14)
    ax.set_ylabel("Average Makespan", fontsize=12)
    ax.set_xlabel("Scheduler Configuration", fontsize=12)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Relative Makespan", fontsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved performance histogram to {output_path}")


def main():
    # #record start time
    # start_time = time.perf_counter()

    #run_example()
    run_experiment()
    #analyze_results()

    # #record end time
    # end_time = time.perf_counter()
    # elapsed = end_time - start_time
    # print(f"Execution time: {elapsed:.2f} seconds")
    #plot_performance_histogram(
    #csv_path=thisdir/"parametric_makespan_comparison.csv",
    #output_path=thisdir/"parametric_performance_histogram.png"
    #)
    
    workflow = get_wfcommons_instance(recipe_name="montage", ccr=1)


if __name__ == "__main__":
    main()