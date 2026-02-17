import random
from typing import Tuple, Dict, List
import networkx as nx
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from saga.schedulers.cpop import CpopScheduler, upward_rank
from saga.schedulers.heft import HeftScheduler
from saga import NetworkEdge, NetworkNode, Schedule, Scheduler, ScheduledTask, TaskGraph, Network
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from saga.utils.random_graphs import (
    get_branching_dag, get_chain_dag, get_diamond_dag, get_fork_dag,
    get_network, add_random_weights
)
from saga.schedulers.data.wfcommons import get_workflows, get_networks
from copy import deepcopy
import numpy as np
import networkx as nx 
import logging
from itertools import product

thisdir = pathlib.Path(__file__).parent.absolute()

def get_random_instance(ccr: float, levels: int, branching_factor: int, num_nodes: int) -> Tuple[Network, TaskGraph]:
    network = get_network(num_nodes=num_nodes)
    task_graph = get_branching_dag(
        levels=levels,
        branching_factor=branching_factor
    )
    # add_random_weights(task_graph)
    network = to_ccr(task_graph, network, ccr)

    return network, task_graph

def to_ccr(task_graph: TaskGraph, 
           network: Network,
           ccr: float) -> Network:
    """Get the network graph to run the task graph on with the given CCR.

    CCR is the communication to computation ratio. The higher the CCR, the more
    communication-heavy the task graph is. The lower the CCR, the more computation-heavy
    the task graph is.

    Args:
        task_graph (TaskGraph): The task graph.
        network (Network): The network graph.
        ccr (float): The communication to computation ratio.

    Returns:
        Network: The network graph.
    """
    network = deepcopy(network)
    mean_task_weight = np.mean([
        task.cost for task in task_graph.tasks
    ])
    mean_dependency_weight = np.mean([
        task_graph_dependency.size for task_graph_dependency in task_graph.dependencies
    ])
    mean_node_weight = np.mean([
        node.speed for node in network.nodes
    ])
    
    link_strength = (mean_dependency_weight / ccr) / (mean_task_weight / mean_node_weight)

    # for edge in network.edges:
    #     if edge[0] == edge[1]:
    #         network.edges[edge]["weight"] = 1e9
    #     else:
    #         network.edges[edge]["weight"] = link_strength

    new_network = Network(
        nodes=[
            NetworkNode(
                name=node.name,
                speed=node.speed
            ) for node in network.nodes
        ],
        edges=[
            NetworkEdge(
                source=edge.source,
                target=edge.target,
                speed=link_strength if edge.source != edge.target else 1e9)
            for edge in network.edges
        ]  
    )

    return new_network

def run_experiment():
    num_instances = 80
    ccr_values = [1/10, 5, 10]
    duplicate_factors = [1,2]
    all_num_nodes = [4, 8]
    all_levels = [2,3]
    all_branching_factors = [2,3]

    schedulers = {
        "HEFT": HeftScheduler,
        "CPoP": CpopScheduler 
    }
    
    # Calculate total iterations for progress bar
    total_iterations = (
        num_instances * 
        len(ccr_values) * 
        len(list(product(all_num_nodes, all_levels, all_branching_factors))) *
        len(duplicate_factors) * 
        len(schedulers)
    )
    
    rows = []
    with tqdm(total=total_iterations, desc="Running experiments") as pbar:
        for i in range(num_instances):
            for ccr in ccr_values:
                for num_nodes, levels, branching_factor in product(all_num_nodes, all_levels, all_branching_factors):
                    network, task_graph = get_random_instance(
                        ccr,
                        levels=levels,
                        branching_factor=branching_factor,
                        num_nodes=num_nodes
                    )
                    for dup_factor in duplicate_factors:
                        for scheduler_name, SchedulerClass in schedulers.items():
                            scheduler = SchedulerClass(duplication_factor=dup_factor)
                            schedule: Schedule = scheduler.schedule(network, task_graph)
                            # schedule_order = schedule.get_schedule_order()
                            # makespan = get_makespan(schedule = schedule) 
                            makespan = schedule.makespan
                            rows.append([i, ccr, num_nodes, levels, branching_factor, scheduler_name, dup_factor, makespan])
                            pbar.update(1)

    df = pd.DataFrame(rows, columns=["Instance", "CCR", "Num Nodes", "Levels", "Branching Factor", "Scheduler", "Dup Factor", "Makespan"])
    df.to_csv(thisdir / "results.csv", index=False)
    print(f"Results saved to {thisdir / 'results.csv'}")


def run_wfcommons_experiment():
    """Run experiments using real workflow traces from WfCommons."""
    num_workflows = 10  # Number of workflows to generate per recipe
    num_networks = 10   # Number of networks to generate
    duplicate_factors = [1, 2]
    
    # Available workflow recipes
    workflow_recipes = ['montage', 'epigenomics', 'seismology']
    
    schedulers = {
        "HEFT": HeftScheduler,
        # "CPoP": CpopScheduler
    }
    
    print("Generating networks from WfCommons data...")
    networks = get_networks(num=num_networks, cloud_name="chameleon", network_speed=100)
    
    # Calculate total iterations for progress bar
    total_iterations = (
        len(workflow_recipes) * 
        num_workflows * 
        len(networks) * 
        len(duplicate_factors) * 
        len(schedulers)
    )
    
    rows = []
    with tqdm(total=total_iterations, desc="Running WfCommons experiments") as pbar:
        for recipe_name in workflow_recipes:
            print(f"\nGenerating {num_workflows} workflows from {recipe_name} recipe...")
            task_graphs = get_workflows(num=num_workflows, recipe_name=recipe_name)
            
            for i, task_graph in enumerate(task_graphs):
                for j, network in enumerate(networks):
                    for dup_factor in duplicate_factors:
                        for scheduler_name, Scheduler in schedulers.items():
                            scheduler = Scheduler(duplication_factor=dup_factor)
                            schedule = scheduler.schedule(network, task_graph)
                            makespan = schedule.makespan
                            
                            rows.append([
                                recipe_name,
                                i,  # workflow instance
                                j,  # network instance
                                len(task_graph.tasks),  # num tasks
                                len(network.nodes),  # num processors
                                scheduler_name,
                                dup_factor,
                                makespan
                            ])
                            pbar.update(1)
    
    df = pd.DataFrame(rows, columns=[
        "Recipe", 
        "Workflow Instance", 
        "Network Instance",
        "Num Tasks",
        "Num Processors",
        "Scheduler", 
        "Dup Factor", 
        "Makespan"
    ])
    df.to_csv(thisdir / "results_wfcommons.csv", index=False)
    print(f"\nWfCommons results saved to {thisdir / 'results_wfcommons.csv'}")


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
    
def draw_schedule(schedule: Schedule, name: str, xmax: float = None):
    ax: plt.Axes = draw_gantt(schedule, use_latex=True, xmax=xmax)
    fig = ax.get_figure()
    fig.savefig(str(thisdir / f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(str(thisdir / f'{name}.pdf'), bbox_inches='tight') 


def main():
    run_experiment()
    # run_wfcommons_experiment()

if __name__ == '__main__':
    main()