import random
from typing import Tuple
import networkx as nx
import pandas as pd
from saga.schedulers.cpop import CpopScheduler
from saga.schedulers.heft import HeftScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from saga.utils.random_graphs import (
    get_branching_dag, get_chain_dag, get_diamond_dag, get_fork_dag,
    get_network, add_random_weights
)
import pathlib
import plotly.express as px


thisdir = pathlib.Path(__file__).parent.absolute()


def get_random_instance() -> Tuple[nx.Graph, nx.DiGraph]:
    network = get_network(num_nodes=random.randint(3, 8))
    
    choice = random.choice(["chain", "fork", "diamond", "branching"])
    if choice == "chain":
        task_graph = get_chain_dag(
            num_nodes=random.randint(5, 10)
        )
    elif choice == "fork":
        task_graph = get_fork_dag()
    elif choice == "diamond":
        task_graph = get_diamond_dag()
    elif choice == "branching":
        task_graph = get_branching_dag(
            levels=random.randint(2,4),
            branching_factor=random.randint(2,4)
        )

    add_random_weights(network)
    add_random_weights(task_graph)

    return network, task_graph

def main():
    num_instances = 100

    cpop_scheduler = CpopScheduler(duplicate_factor=1)
    cpop_scheduler_dup = CpopScheduler(duplicate_factor=2)

    heft_scheduler = HeftScheduler(duplicate_factor=1)
    heft_scheduler_dup = HeftScheduler(duplicate_factor=2)

    rows = []
    for i in range(num_instances):
        network, task_graph = get_random_instance()
        cpop_schedule = cpop_scheduler.schedule(network, task_graph)
        heft_schedule = heft_scheduler.schedule(network, task_graph)
        cpop_duplicate_schedule = cpop_scheduler_dup.schedule(network, task_graph)
        heft_duplicate_schedule = heft_scheduler_dup.schedule(network, task_graph)

        #getting the makespan
        cpop_makespan = max([0 if not tasks else tasks[-1].end for tasks in cpop_schedule.values()])
        heft_makespan = max([0 if not tasks else tasks[-1].end for tasks in heft_schedule.values()])
        cpop_duplicate_makespan = max([0 if not tasks else tasks[-1].end for tasks in cpop_duplicate_schedule.values()])
        heft__duplicate_makespan = max([0 if not tasks else tasks[-1].end for tasks in heft_duplicate_schedule.values()])

        rows.append([i, cpop_makespan, heft_makespan, cpop_duplicate_makespan, heft__duplicate_makespan])

    df = pd.DataFrame(rows, columns=["Instance", "CPoP", "HEFT", "CPoPDup", "HEFTDup"])

    fig = px.box(
        df,
        y=["CPoP", "HEFT", "CPoPDup", "HEFTDup"],
        template="simple_white",
        title="Makespan Comparison for duplication",
        labels={"variable": "Scheduler", "value": "Makespan"},
    )
    fig.write_image(thisdir / "results.pdf")
    fig.write_image(thisdir / "results.png")

if __name__ == '__main__':
    main()

