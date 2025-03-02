from copy import deepcopy
from itertools import product
from saga.schedulers import HeftScheduler
from saga.scheduler import Scheduler, Task
# from saga.utils.random_variable import RandomVariable

import pygraphviz as pgv
from typing import Dict, Hashable, List, Tuple
import networkx as nx
import matplotlib.cm as cm
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from saga.utils.draw import gradient_heatmap

import pathlib


thisdir = pathlib.Path(__file__).resolve().parent


def schedule_estimate_to_actual(network: nx.Graph,
                                task_graph: nx.DiGraph,
                                schedule_estimate: Dict[Hashable, List[Task]]) -> Dict[Hashable, List[Task]]:
    """
    Converts an 'estimated' schedule (using mean or estimate-based runtimes)
    into an 'actual' schedule, using hidden static true values stored in the
    node and task attributes. 

    Assumptions:
      - network.nodes[node]['weight_actual'] holds the node's true speed.
      - task_graph.nodes[task]['weight_actual'] holds the task's true cost.
      - We preserve the order of tasks within each node exactly as in schedule_estimate.
      - Each new task on a node starts when the previous task on that node actually finishes.
        (No overlap on a single node.)

    Args:
        network (nx.Graph): The network graph, with per-node 'weight_actual'.
        task_graph (nx.DiGraph): The task graph, with per-task 'weight_actual'.
        schedule_estimate (Dict[Hashable, List[Task]]): The estimated schedule 
            as a dict: node -> list of Task objects in the order they are 
            planned to run.

    Returns:
        Dict[Hashable, List[Task]]: The actual schedule, with tasks' start/end
            times reflecting real runtimes.
    """

    #intitiate schedule_actual
    schedule_actual: Dict[Hashable, List[Task]] = {
        node: [] for node in network.nodes #creating an empty schedule
    }
    #creating a dict of tasks and filling it with the tasks in estimate schedule I believe this is done simply 
    #to make it easier to access our tasks, which is something he said he does throughout the code
    tasks_estimate: Dict[str, Task] = { 
        task.name: task
        for node, tasks in schedule_estimate.items()
        for task in tasks
    }
    tasks_actual: Dict[str, Task] = {} #Creating an empty dict to store our actual tasks once they are completed

    task_order = sorted(task_graph.nodes, key=lambda task: tasks_estimate[task].start) #ordering our nodes based on their estimated start times
    for task_name in task_order:#looping through our sorted tasks 
        start_time = 0 #setting start time to be used later. Start time resets with each task due to how it is later calculated
        task_node = tasks_estimate[task_name].node#pulling out the current task from our estimated and setting it to our current
        for parent_name in task_graph.predecessors(task_name): #loop to get end times of previous, aka parent tasks
            end_time = tasks_actual[parent_name].end #setting end time to the end of parent task
            parent_node = tasks_actual[parent_name].node #setting parent node 
            #getting the communication time for calculating time values by getting communication time between parent and current task
            comm_time = task_graph.edges[(parent_name, task_name)]['weight_actual'] / network.edges[(task_node, parent_node)]['weight_actual']
            #setting start time to either whatever the current highest start time is, or to the end time of current parent_node plus its com time to
            #current task node. This is basically making sure we dont try and start before all parent nodes have finished and communicated
            start_time = max(start_time, end_time + comm_time)

        #If current task is already in actual, set start time to either current calculated, or the end time in the last thing that actually finished
        if schedule_actual[task_node]: 
            start_time = max(start_time, schedule_actual[task_node][-1].end)#

        #calculating runtime of current task 
        runtime = task_graph.nodes[task_name]["weight_actual"] / network.nodes[task_node]["weight_actual"]
        new_task = Task( #creating a "new" task with all of our calculated attributes 
            node=task_node,
            name=task_name,
            start=start_time,
            end=start_time + runtime
        )
        schedule_actual[task_node].append(new_task) #putting into our actual scheduler 
        tasks_actual[task_name] = new_task #putting into our dict of actual tasks 

    return schedule_actual


class OnlineHeftScheduler(Scheduler):
    def __init__(self):
        self.heft_scheduler = HeftScheduler()

    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 do_print: bool = False) -> Dict[Hashable, List[Task]]:
        """
        A 'live' scheduling loop using HEFT in a dynamic manner.
        We assume each node/task has 'weight_actual' and 'weight_estimate' attributes.
        
        1) No longer needed
        2) Calls the standard HEFT for an 'estimated' schedule.
        3) Converts that schedule to 'actual' times using schedule_estimate_to_actual.
        4) Commits the earliest-finishing new task to schedule_actual.
        5) Repeats until all tasks are scheduled.        

        comp_schedule = {
        "Node1": [Task(node="Node1", name="TaskA", start=0, end=5)],
        "Node2": [Task(node="Node2", name="TaskB", start=3, end=8)],
        """
        schedule_actual: Dict[Hashable, List[Task]] = {
            node: [] for node in network.nodes
        }
        tasks_actual: Dict[str, Task] = {}
        current_time = 0

        while len(tasks_actual) < len(task_graph.nodes):
            #------------------------------------------Step 2: Get next task to finish based on *actual* execution time
            
            schedule_actual_hypothetical = schedule_estimate_to_actual(
                network,
                task_graph,
                self.heft_scheduler.schedule(network, task_graph, schedule_actual, min_start_time=current_time)
            )

            tasks: List[Task] = sorted([task for node_tasks in schedule_actual_hypothetical.values() for task in node_tasks], key=lambda x: x.start)
            next_task: Task = min(
                [task for task in tasks if task.name not in tasks_actual],
                key=lambda x: x.end
            )
            current_time = next_task.end
            if do_print:
                print(f"Next task to finish: {next_task.name}")
            
            for task in tasks:
                if task.start < next_task.end and task.name not in tasks_actual:
                    if do_print:
                        print(f"Adding task {task.name} to actual schedule")
                    schedule_actual[task.node].append(task)
                    tasks_actual[task.name] = task

        return schedule_actual
    
from saga.utils.random_graphs import get_branching_dag, get_network, get_diamond_dag, get_chain_dag, get_fork_dag
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def get_instance() -> Tuple[nx.Graph, nx.DiGraph]:
    # Create a random network
    network = get_network(num_nodes=4)
    # Create a random task graph
    #task_graph = get_fork_dag()
    #task_graph = get_branching_dag(levels=3, branching_factor=5)
    #task_graph = get_chain_dag(num_nodes=10)
    task_graph = get_diamond_dag()
    # network = add_rv_weights(network)
    # task_graph = add_rv_weights(task_graph)

    min_mean = 3
    max_mean = 10
    min_std = 1/2
    max_std = 1

    for node in network.nodes:
        mean = np.random.uniform(min_mean, max_mean)
        std = np.random.uniform(min_std, max_std)
        network.nodes[node]["weight_estimate"] = mean
        network.nodes[node]["weight_actual"] = max(1e-9, np.random.normal(mean, std))
        network.nodes[node]["weight"] = network.nodes[node]["weight_estimate"]

    for (u, v) in network.edges:
        if u == v:
            network.edges[u, v]["weight_estimate"] = 1e9
            network.edges[u, v]["weight_actual"] = 1e9

        mean = np.random.uniform(min_mean, max_mean)
        std = np.random.uniform(min_std, max_std)
        network.edges[u, v]["weight_estimate"] = mean
        network.edges[u, v]["weight_actual"] = max(1e-9, np.random.normal(mean, std))
        network.edges[u, v]["weight"] = network.edges[u, v]["weight_estimate"]

    for task in task_graph.nodes:
        mean = np.random.uniform(min_mean, max_mean)
        std = np.random.uniform(min_std, max_std)
        task_graph.nodes[task]["weight_estimate"] = mean
        task_graph.nodes[task]["weight_actual"] = max(1e-9, np.random.normal(mean, std))
        task_graph.nodes[task]["weight"] = task_graph.nodes[task]["weight_estimate"]
        

    for (src, dst) in task_graph.edges:
        mean = np.random.uniform(min_mean, max_mean)
        std = np.random.uniform(min_std, max_std)
        task_graph.edges[src, dst]["weight_estimate"] = mean
        task_graph.edges[src, dst]["weight_actual"] = max(1e-9, np.random.normal(mean, std))
        task_graph.edges[src, dst]["weight"] = task_graph.edges[src, dst]["weight_estimate"]

    return network, task_graph

def get_instance_custom(levels: int, branching_factor: int) -> Tuple[nx.Graph, nx.DiGraph]:
    """
    Create a network and a branching task graph with the specified number
    of levels and branching factor.
    """
    network = get_network(num_nodes=4)
    task_graph = get_branching_dag(levels=levels, branching_factor=branching_factor)

    #Make graph for varying these
    min_mean = 3
    max_mean = 10
    min_std = 0.5
    max_std = 1.0

    for node in network.nodes:
        mean = np.random.uniform(min_mean, max_mean)
        std = np.random.uniform(min_std, max_std)
        network.nodes[node]["weight_estimate"] = mean
        network.nodes[node]["weight_actual"] = max(1e-9, np.random.normal(mean, std))
        network.nodes[node]["weight"] = network.nodes[node]["weight_estimate"]

    for (u, v) in network.edges:
        if u == v:
            network.edges[u, v]["weight_estimate"] = 1e9
            network.edges[u, v]["weight_actual"] = 1e9
        mean = np.random.uniform(min_mean, max_mean)
        std = np.random.uniform(min_std, max_std)
        network.edges[u, v]["weight_estimate"] = mean
        network.edges[u, v]["weight_actual"] = max(1e-9, np.random.normal(mean, std))
        network.edges[u, v]["weight"] = network.edges[u, v]["weight_estimate"]

    for task in task_graph.nodes:
        mean = np.random.uniform(min_mean, max_mean)
        std = np.random.uniform(min_std, max_std)
        task_graph.nodes[task]["weight_estimate"] = mean
        task_graph.nodes[task]["weight_actual"] = max(1e-9, np.random.normal(mean, std))
        task_graph.nodes[task]["weight"] = task_graph.nodes[task]["weight_estimate"]

    for (src, dst) in task_graph.edges:
        mean = np.random.uniform(min_mean, max_mean)
        std = np.random.uniform(min_std, max_std)
        task_graph.edges[src, dst]["weight_estimate"] = mean
        task_graph.edges[src, dst]["weight_actual"] = max(1e-9, np.random.normal(mean, std))
        task_graph.edges[src, dst]["weight"] = task_graph.edges[src, dst]["weight_estimate"]

    return network, task_graph

def get_offline_instance(network: nx.Graph, task_graph: nx.DiGraph) -> Tuple[nx.Graph, nx.DiGraph]:
    # replace weights with actual weights
    for node in network.nodes:
        network.nodes[node]["weight"] = network.nodes[node]["weight_actual"]

    for (u, v) in network.edges:
        network.edges[u, v]["weight"] = network.edges[u, v]["weight_actual"]

    for task in task_graph.nodes:
        task_graph.nodes[task]["weight"] = task_graph.nodes[task]["weight_actual"]

    for (src, dst) in task_graph.edges:
        task_graph.edges[src, dst]["weight"] = task_graph.edges[src, dst]["weight_actual"]

    return network, task_graph

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
    
    link_strength = mean_dependency_weight / (ccr * mean_task_weight)

    for edge in network.edges:
        if edge[0] == edge[1]:
            network.edges[edge]["weight"] = 1e9
        else:
            network.edges[edge]["weight"] = link_strength

    return network

def run_sample(params: Tuple[float, int, int, int]) -> List[Tuple[float, int, int, int, float, str]]:
    """
    Runs one experiment sample for a given set of parameters.
    
    Args:
        params: A tuple (ccr, levels, branching_factor, sample_index)
    
    Returns:
        A list of three tuples, one for each scheduler variant:
            (ccr, levels, branching_factor, sample_index, makespan, scheduler_name)
    """
    
    ccr, levels, bf, sample_index = params
    scheduler = HeftScheduler()
    scheduler_online = OnlineHeftScheduler()
    network, task_graph = get_instance_custom(levels, bf)
    network_ccr = to_ccr(task_graph, network, ccr)
    print(f"Running experiment (CCR={ccr}, levels={levels}, branching_factor={bf}, sample={sample_index})")
    
    # Run standard HEFT (Naive Online HEFT)
    schedule_custom = scheduler.schedule(network_ccr, task_graph)
    schedule_actual_custom = schedule_estimate_to_actual(network_ccr, task_graph, schedule_custom)
    makespan_custom = max(task.end for node_tasks in schedule_actual_custom.values() for task in node_tasks)
    
    # Run Online HEFT
    schedule_online_custom = scheduler_online.schedule(network_ccr, task_graph)
    makespan_online_custom = max(task.end for node_tasks in schedule_online_custom.values() for task in node_tasks)
    
    # Run Offline HEFT (knows actual weights)
    network_offline, task_graph_offline = get_offline_instance(network_ccr, task_graph)
    schedule_offline = scheduler.schedule(network_offline, task_graph_offline)
    makespan_offline = max(task.end for node_tasks in schedule_offline.values() for task in node_tasks)
    
    return [
        (ccr, levels, bf, sample_index, makespan_custom, "Naive Online HEFT"),
        (ccr, levels, bf, sample_index, makespan_online_custom, "Online HEFT"),
        (ccr, levels, bf, sample_index, makespan_offline, "Offline HEFT")
    ]

def run_experiment():
    scheduler = HeftScheduler()
    scheduler_online = OnlineHeftScheduler()

    cores = 4
    n_samples = 50
    experiments = []
    ccrs = [1/5, 1/2, 1, 2, 5]
    levels_range = [1, 2, 3, 4]         # x axis: Levels
    branching_range = [1, 2, 3, 4]      # y axis: Branching Factor
    all_params = list(product(ccrs, levels_range, branching_range, range(n_samples)))
    #total_experiments = len(all_params) * 3  # three experiments per instance
    
    #total_experiments = len(ccrs) * len(levels_range) * len(branching_range) * n_samples * 3
    '''
    for ccr in ccrs:
        for levels, bf in product(levels_range, branching_range):
            for i in range(n_samples):
                network, task_graph = get_instance_custom(levels, bf)
                network_ccr = to_ccr(task_graph, network, ccr)
                print(
                    f"Running experiment {len(experiments) + 1}/{total_experiments}"
                    f"(CCR={ccr}, levels={levels}, branching_factor={bf}, sample={i})"
                    f" - {len(experiments) / total_experiments * 100:.2f}%"
                )
    '''

    #multiprocessing
    with Pool(processes=cores) as pool:
        results = pool.map(run_sample, all_params)
    
    # Flatten results
    for res in results:
        experiments.extend(res)
    df = pd.DataFrame(experiments, columns=["ccr", "levels", "branching_factor", "instance", "makespan", "scheduler"])
    df.to_csv(thisdir / "makespan_experiments.csv")

def analyze_results():
    df = pd.read_csv(thisdir / "makespan_experiments.csv", index_col=0)
    # get makespan ratio of (makespan_online / makespan_offline)
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
                ax=ax
            )
        
        fig.tight_layout()
        fig.savefig(thisdir / f"makespan_heatmap_ccr_{ccr}.png")

def main():
    #record start time
    start_time = time.perf_counter()

    #run_experiment()
    analyze_results()

    #record end time
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()