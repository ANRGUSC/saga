import random
from typing import Tuple, Dict, List
import networkx as nx
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

from saga.schedulers.cpop import CpopScheduler
from saga.schedulers.heft import HeftScheduler
from saga.scheduler import Task
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from saga.utils.random_graphs import (
    get_branching_dag, get_chain_dag, get_diamond_dag, get_fork_dag,
    get_network, add_random_weights
)
import pathlib
import plotly.express as px
from copy import deepcopy
import numpy as np
import networkx as nx 
import logging

thisdir = pathlib.Path(__file__).parent.absolute()


def get_random_instance(ccr: float) -> Tuple[nx.Graph, nx.DiGraph]:
    network = get_network(num_nodes=2)
    task_graph = get_branching_dag(
        levels=3,
        branching_factor=2
    )
    # task_graph = get_fork_dag()
    # task_graph = get_diamond_dag()
    add_random_weights(network, weight_range=(1,1))
    add_random_weights(task_graph)
    network = to_ccr(task_graph, network, ccr)

    return network, task_graph

def get_makespan(schedule):
    return max(
        (task.end for tasks in schedule.values() for task in tasks),
        default=0
    )

def to_ccr(task_graph: nx.DiGraph, 
           network: nx.Graph,
           ccr: float) -> nx.Graph:
    """Get the network graph to run the task graph on with the given CCR.

    CCR is the communication to computation ratio. The higher the CCR, the more
    communication-heavy the task graph is. The lower the CCR, the more computation-heavy
    the task graph is.

    Args:
        task_graph (nx.DiGraph): The task graph.
        network (nx.Graph): The network graph.
        ccr (float): The communication to computation ratio.

    Returns:
        nx.Graph: The network graph.
    """
    network = deepcopy(network)
    mean_task_weight = np.mean([
        task_graph.nodes[node]["weight"]
        for node in task_graph.nodes
    ])
    mean_dependency_weight = np.mean([
        task_graph.edges[edge]["weight"]
        for edge in task_graph.edges
    ])
    mean_node_weight = np.mean([
        network.edges[edge]["weight"]
        for edge in network.edges
        if edge[0] != edge[1]
    ])
    
    link_strength = (mean_dependency_weight / ccr) / (mean_task_weight / mean_node_weight)

    for edge in network.edges:
        if edge[0] == edge[1]:
            network.edges[edge]["weight"] = 1e9
        else:
            network.edges[edge]["weight"] = link_strength

    return network

def remove_outliers_iqr_grouped(df: pd.DataFrame, group_cols, value_col: str, k: float = 1.5) -> pd.DataFrame:
    """
    Remove rows whose value_col is an outlier within its group defined by group_cols,
    using the IQR rule with multiplier k.
    """
    df2 = df.copy()
    mask = pd.Series(True, index=df2.index)
    for name, group in df2.groupby(group_cols):
        q1 = group[value_col].quantile(0.25)
        q3 = group[value_col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        mask.loc[group.index] = group[value_col].between(lower, upper)
    return df2[mask]

def draw_one():
    ccr = 10
    
    network, task_graph = get_random_instance(ccr)
    draw_instance(network=network, task_graph=task_graph)

    scheduler = CpopScheduler(duplicate_factor=1)
    schedule = scheduler.schedule(network, task_graph)
    makespan = get_makespan(schedule = schedule) 
    draw_schedule(schedule, 'cpop_schedule', xmax = makespan)

    scheduler_dup2 = CpopScheduler(duplicate_factor=2)
    schedule_dup2 = scheduler_dup2.schedule(network, task_graph)
    makespan_dup2 = get_makespan(schedule = schedule_dup2) 
    draw_schedule(schedule_dup2, 'cpop_schedule_dup2', xmax = makespan_dup2)


def run_experiment():
    num_instances = 10
    ccr_values = [1/2, 1, 2, 5, 10]
    duplicate_factors = [1, 2]

    schedulers = {
        "HEFT": HeftScheduler,
        "CPoP": CpopScheduler
    }
    
    rows = []
    for i in range(num_instances):
        for ccr in ccr_values:
            network, task_graph = get_random_instance(ccr)
            for dup_factor in duplicate_factors:
                for scheduler_name, Scheduler in schedulers.items():
                    scheduler = Scheduler(duplicate_factor=dup_factor)
                    schedule = scheduler.schedule(network, task_graph)
                    makespan = get_makespan(schedule = schedule) 
                    rows.append([i, ccr, scheduler_name, str(dup_factor), makespan])

    df = pd.DataFrame(rows, columns=["Instance", "CCR", "Scheduler", "Dup Factor", "Makespan"])

    df = remove_outliers_iqr_grouped(
        df, 
        group_cols=["Scheduler", "Dup Factor", "CCR"],
        value_col="Makespan"
    )


    fig = px.box(
        df,
        x="Scheduler",
        y="Makespan",
        color="Dup Factor",
        facet_col="CCR",
        facet_col_wrap=3,
        template="simple_white",
        title="Makespan Comparison for duplication",
        labels={"variable": "Scheduler", "value": "Makespan"},
        points=None
    )
    fig.write_image(thisdir / "results.pdf")
    fig.write_image(thisdir / "results.png")


def draw_instance(network: nx.Graph, task_graph: nx.DiGraph):
    logging.basicConfig(level=logging.INFO)
    ax: plt.Axes = draw_task_graph(task_graph, use_latex=True)
    fig = ax.get_figure()
    fig.savefig(str(thisdir / 'task_graph.png'), dpi=300, bbox_inches='tight')
    fig.savefig(str(thisdir / 'task_graph.pdf'), bbox_inches='tight')

    ax = draw_network(network, draw_colors=False, use_latex=True)
    fig = ax.get_figure()
    fig.savefig(str(thisdir / 'network.png'), dpi=300, bbox_inches='tight')
    fig.savefig(str(thisdir / 'network.pdf'), bbox_inches='tight')
    
def draw_schedule(schedule: Dict[str, List[Task]], name: str, xmax: float = None):
    ax: plt.Axes = draw_gantt(schedule, use_latex=True, xmax=xmax)
    fig = ax.get_figure()
    fig.savefig(str(thisdir / f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(str(thisdir / f'{name}.pdf'), bbox_inches='tight') 


def main():
    draw_one()
    # run_experiment()

if __name__ == '__main__':
    main()

