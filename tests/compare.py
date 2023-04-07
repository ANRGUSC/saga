from typing import Dict
import numpy as np
from saga.base import Scheduler
from saga.stochastic.mean_heft import MeanHeftScheduler
from saga.stochastic.sheft import SheftScheduler
from saga.stochastic.improved_sheft import ImprovedSheftScheduler
from saga.stochastic.stoch_heft import StochHeftScheduler
from saga.utils.simulator import Simulator
import pandas as pd
import pathlib

from saga.utils.random_graphs import (
    get_diamond_dag, get_chain_dag, get_fork_dag, get_branching_dag, 
    get_network, add_random_weights, add_rv_weights
)

thisdir = pathlib.Path(__file__).parent.absolute()

def run():
    # logging.basicConfig(level=logging.INFO)

    schedulers: Dict[str, Scheduler] = {
        "SHEFT": SheftScheduler(),
        "Improved SHEFT": ImprovedSheftScheduler(),
        "Stochastic HEFT": StochHeftScheduler(),
        "Mean HEFT": MeanHeftScheduler(),
    }

    num_samples = 20

    rows = []
    for sample_i in range(num_samples):
        task_graphs = {
            "Diamond": add_rv_weights(get_diamond_dag()),
            "Chain": add_rv_weights(get_chain_dag()),
            "Fork": add_rv_weights(get_fork_dag()),
            "Branching": add_rv_weights(get_branching_dag()),
        }
        network = add_rv_weights(get_network())
        for task_graph_name, task_graph in task_graphs.items():
            for scheduler_name, scheduler in schedulers.items():
                schedule = scheduler.schedule(network, task_graph)
                simulator = Simulator(network, task_graph, schedule)
                schedules = simulator.run(num_simulations=100)
                makespans = [
                    max([task.end for _, tasks in schedule.items() for task in tasks])
                    for schedule in schedules
                ]
                avg_make_span = np.mean(makespans)
                print(f"Sample {sample_i}: {task_graph_name} {scheduler_name} {avg_make_span}")
                rows.append([sample_i, task_graph_name, scheduler_name, avg_make_span])

    df = pd.DataFrame(rows, columns=["sample", "task_graph", "scheduler", "makespan"])
    path = thisdir.joinpath("output", "stochastic_comparison", "results.csv")
    df.to_csv(path, index=False)

import plotly.express as px
def analyze():
    df = pd.read_csv("results.csv")
    # get mean and std
    df = df.groupby(["task_graph", "scheduler"]).agg({"makespan": ["mean", "std"]})
    print(df)

    # visualize as a bar with categories
    df = df.reset_index()
    df.columns = ["task_graph", "scheduler", "mean", "std"]
    fig = px.bar(
        df, x="task_graph", y="mean", error_y="std", 
        color="scheduler", barmode="group", 
        title="Makespan Comparison",
        labels={"task_graph": "Task Graph", "scheduler": "Scheduler", "mean": "Makespan"}
    )
    # center title
    fig.update_layout(title_x=0.5)
    path = thisdir.joinpath("output", "stochastic_comparison", "results.html")
    fig.write_html(path)

def main():
    run()
    analyze()

if __name__ == "__main__":
    main()