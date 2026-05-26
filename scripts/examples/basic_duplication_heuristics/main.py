from saga.utils.random_graphs import (
    get_network, 
    get_branching_dag,
    get_chain_dag,
    get_diamond_dag,
    get_fork_dag
)
from saga.schedulers import HeftScheduler, CpopScheduler
import saga.schedulers.heft as heft_mod
import saga.schedulers.cpop as cpop_mod

from saga.utils.duplication_heuristics import task_score
from saga import TaskGraph

import pandas as pd
import pathlib
import matplotlib.pyplot as plt

thisdir = pathlib.Path(__file__).parent.resolve()
ALLOWED_DUPLICATES: set[str] = set()

def select_should_duplicate(task_name: str, task_graph: TaskGraph, network) -> bool:
    return task_name in ALLOWED_DUPLICATES

def get_scoring_tasks(task_graph: TaskGraph, n: int, mode: str = "top-n") -> list[str]:
    scored_tasks = []

    for task in task_graph.tasks:
        task_name = task.name

        # ignore fake nodes (super source/sinks)
        if task_name.startswith("__super_"):
            continue

        score = task_score(task_name, task_graph)
        scored_tasks.append((task_name, score))
    
    reverse = mode == "top-n"
    scored_tasks.sort(key=lambda item: item[1], reverse=reverse)
    # return only task names 
    return [task_name for task_name, score in scored_tasks[:n]]



def main():
    global ALLOWED_DUPLICATES

    num_experiments = 100
    max_top_n = 5

    scheduler_classes = [
		("heft", HeftScheduler),
		("cpop", CpopScheduler),
	]

    heft_mod.should_duplicate = select_should_duplicate
    cpop_mod.should_duplicate = select_should_duplicate

    data = []
    for i in range(num_experiments):
        network = get_network()
        task_graph = get_branching_dag()
        for scheduler_name, scheduler_class in scheduler_classes:
            for mode in ["top-n", "bottom-n"]:
                all_scoring_tasks = get_scoring_tasks(task_graph, max_top_n, mode)
                for n in range(max_top_n + 1):
                    selected_tasks = all_scoring_tasks[:n]
                    ALLOWED_DUPLICATES = set(selected_tasks)
                    scheduler = scheduler_class(duplication_factor=2) 
                    schedule = scheduler.schedule(network, task_graph)

                    data.append({
                        "experiment_num": i,
                        "scheduler_name": scheduler_name,
                        "mode": mode,
                        "n": n,
                        "duplicated_tasks": ",".join(selected_tasks),
                        "makespan": schedule.makespan
                    })

    df = pd.DataFrame(data)
    df.to_csv(thisdir / "data.csv", index=False)

    for scheduler_name in df["scheduler_name"].unique():
        scheduler_df = df[df["scheduler_name"] == scheduler_name]
        # for each mode and n, calculate avg makespan & standard deviation
        summary = scheduler_df.groupby(["mode", "n"])["makespan"].agg(["mean", "std"]).reset_index()
        print(summary)
        plt.figure(figsize=(8, 5))

        for mode in summary["mode"].unique():
            mode_df = summary[summary["mode"] == mode]
            # x-axis = num of duplicated_tasks, y-axis = avg makespan, err_bars = std
            plt.errorbar(
                mode_df["n"], 
                mode_df["mean"], 
                yerr=mode_df["std"], #["sem"]
                marker="o", 
                capsize=5, 
                label=mode
            )
        
        plt.xlabel("N duplicated tasks")
        plt.ylabel("Average Makespan")
        plt.title(f"Average Makespan: Top-N vs Bottom-N Duplicated Tasks ({scheduler_name.upper()})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(thisdir / f"{scheduler_name}_makespan_plot.png")
        plt.close()

if __name__ == "__main__":
    main()