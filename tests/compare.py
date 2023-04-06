from itertools import product
from typing import Callable, Dict, Optional, Tuple, TypeVar
import networkx as nx
import numpy as np
from scipy.stats import norm
from saga.base import Scheduler
from saga.stochastic.mean_heft import MeanHeftScheduler
from saga.stochastic.sheft import SheftScheduler
from saga.stochastic.improved_sheft import ImprovedSheftScheduler
from saga.stochastic.stoch_heft import StochHeftScheduler
from saga.utils.simulator import Simulator
import pandas as pd
import pathlib

from saga.utils.random_variable import RandomVariable

thisdir = pathlib.Path(__file__).parent.absolute()

def get_diamond_dag() -> nx.DiGraph:
    """Returns a diamond DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C", "D"])
    dag.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    return dag

def get_chain_dag() -> nx.DiGraph:
    """Returns a chain DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C", "D"])
    dag.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    return dag

def get_fork_dag() -> nx.DiGraph:
    """Returns a fork DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C", "D", "E", "F"])
    dag.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "E"), ("D", "F"), ("E", "F")])
    return dag


def get_branching_dag(levels: int = 4, branching_factor: int = 2) -> nx.DiGraph:
    G = nx.DiGraph()
    
    node_id = 0
    level_nodes = [node_id]  # Level 0
    G.add_node(node_id)
    node_id += 1
    
    for level in range(1, levels):
        new_level_nodes = []
        
        for parent in level_nodes:
            children = [node_id + i for i in range(branching_factor)]
            
            G.add_edges_from([(parent, child) for child in children])
            new_level_nodes.extend(children)
            node_id += branching_factor
        
        level_nodes = new_level_nodes
    
    # Add destination node
    dst_node = node_id
    G.add_node(dst_node)
    G.add_edges_from([(node, dst_node) for node in level_nodes])

    return G

def get_network() -> nx.Graph:
    """Returns a network."""
    network = nx.Graph()
    network.add_nodes_from(range(4))
    # fully connected
    network.add_edges_from(product(range(4), range(4)))
    return network

# template T nx.DiGraph or nx.Graph
T = TypeVar("T", nx.DiGraph, nx.Graph)
def add_random_weights(graph: T, weight_range: Tuple[float, float] = (1, 10)) -> T:
    """Adds random weights to the DAG."""
    for node in graph.nodes:
        graph.nodes[node]["weight"] = np.random.uniform(*weight_range)
    for edge in graph.edges:
        if not graph.is_directed() and edge[0] == edge[1]:
            graph.edges[edge]["weight"] = 1e9 * weight_range[1] # very large communication speed
        else:
            graph.edges[edge]["weight"] = np.random.uniform(*weight_range)
    return graph

def add_rv_weights(graph: T) -> T:
    """Adds random variable weights to the DAG."""
    def get_rv(weight_range: Tuple[float, float]
               ):
        std = np.random.uniform(1, 3)
        loc = np.random.uniform(weight_range[0] + 3 * std, weight_range[1] - 3 * std)
        x = np.linspace(loc - 3 * std, loc + 3 * std, 1000)
        pdf = norm.pdf(x, loc, std)
        return RandomVariable.from_pdf(x, pdf)
    
    for node in graph.nodes:
        graph.nodes[node]["weight"] = get_rv(weight_range=(20, 50))
    for edge in graph.edges:
        if not graph.is_directed() and edge[0] == edge[1]:
            graph.edges[edge]["weight"] = RandomVariable([1e9], num_samples=1) # very large communication speed
        else:
            graph.edges[edge]["weight"] = get_rv(weight_range=(1, 20))
    return graph

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