import random
from typing import Dict, Tuple
import networkx as nx
import pandas as pd
from saga.scheduler import Scheduler
from saga.schedulers.cpop import CpopScheduler
from saga.schedulers.heft import HeftScheduler
from saga.schedulers.smt import SMTScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from saga.utils.random_graphs import (
    get_branching_dag, get_chain_dag, get_diamond_dag, get_fork_dag,
    get_network, add_random_weights
)
import pathlib
import plotly.express as px


thisdir = pathlib.Path(__file__).parent.absolute()


def get_random_instance() -> Tuple[nx.Graph, nx.DiGraph]:
    network = get_network(num_nodes=4)
    
    choice = random.choice(["chain", "fork", "diamond", "branching"])
    if choice == "chain":
        task_graph = get_chain_dag(
            num_nodes=5
        )
    elif choice == "fork":
        task_graph = get_fork_dag()
    elif choice == "diamond":
        task_graph = get_diamond_dag()
    elif choice == "branching":
        task_graph = get_branching_dag(levels=3, branching_factor=2)

    add_random_weights(network)
    add_random_weights(task_graph)

    return network, task_graph

def main():
    num_instances = 100
    schedulers: Dict[str, Scheduler] = {
        "CPoP": CpopScheduler(),
        "HEFT": HeftScheduler(),
        "SMT": SMTScheduler(epsilon=0.1)
    }

    rows = []
    for i in range(num_instances):
        network, task_graph = get_random_instance()
        makespans = []

        print(f"Num nodes: {network.number_of_nodes()}, Num tasks: {task_graph.number_of_nodes()}")
        for j, (name, scheduler) in enumerate(schedulers.items()):
            print(f"Progress: {j + i * len(schedulers)}/{num_instances * len(schedulers)}")
            schedule = scheduler.schedule(network, task_graph)
            makespan = sum([task.end for tasks in schedule.values() for task in tasks])
            makespans.append(makespan)

        rows.append([i, *makespans])

    df = pd.DataFrame(rows, columns=["Instance", *schedulers.keys()])

    fig = px.box(
        df,
        y=list(schedulers.keys()),
        template="simple_white",
        title="Makespan Comparison",
        labels={"variable": "Scheduler", "value": "Makespan"},
    )
    fig.write_image(thisdir / "results.pdf")
    fig.write_image(thisdir / "results.png")

if __name__ == '__main__':
    main()

