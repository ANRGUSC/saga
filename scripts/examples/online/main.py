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

from saga.schedulers import HeftScheduler
from saga.scheduler import Scheduler, Task
from saga.utils.random_graphs import get_branching_dag, get_network
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

from saga.schedulers.data.wfcommons import get_workflows, get_networks
from saga.utils.random_variable import RandomVariable

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
                 task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
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
        iteration = 0
        while len(tasks_actual) < len(task_graph.nodes):
            iteration += 1
            if iteration > 1000:
                print({t for t in tasks_actual}, set(task_graph.nodes))
                #------------------------------------------Step 2: Get next task to finish based on *actual* execution time
                print("here")
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
            
            for task in tasks:
                if task.start <= current_time and task.name not in tasks_actual:
                    schedule_actual[task.node].append(task)
                    tasks_actual[task.name] = task

        return schedule_actual
    
def get_instance(levels: int, branching_factor: int, ccr: float = 1.0) -> Tuple[nx.Graph, nx.DiGraph]:
    """
    Create a network and a branching task graph with the specified number
    of levels and branching factor.
    """
    network = get_network(num_nodes=4)
    task_graph = get_branching_dag(levels=levels, branching_factor=branching_factor)

    #Make graph for varying these
    min_mean = 3
    max_mean = 20
    min_std = 0.5
    max_std = 3

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
            network.edges[u, v]["weight"] = 1e9
        else:
            mean = np.random.uniform(min_mean, max_mean)
            std = np.random.uniform(min_std, max_std)
            network.edges[u, v]["weight_estimate"] = mean
            network.edges[u, v]["weight_actual"] = max(1e-9, np.random.normal(mean, std))
            network.edges[u, v]["weight"] = network.edges[u, v]["weight_estimate"]

    for task in task_graph.nodes:
        mean = np.random.uniform(min_mean, max_mean) * ccr
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

def get_wfcommons_instance(recipe_name: str, ccr: float):
    workflow = get_workflows(num=1, recipe_name=recipe_name, rv_weights=True)[0]

    # rename weight attribute to weight_rv
    for node in workflow.nodes:
        weight_rv: RandomVariable = workflow.nodes[node]["weight"]
        workflow.nodes[node]["weight_rv"] = weight_rv
        workflow.nodes[node]["weight_estimate"] = weight_rv.mean()
        workflow.nodes[node]["weight_actual"] = weight_rv.sample()
        workflow.nodes[node]["weight"] = workflow.nodes[node]["weight_estimate"]

    for (u, v) in workflow.edges:
        weight_rv: RandomVariable = workflow.edges[u, v]["weight"]
        workflow.edges[u, v]["weight_rv"] = weight_rv
        workflow.edges[u, v]["weight_estimate"] = weight_rv.mean()
        workflow.edges[u, v]["weight_actual"] = weight_rv.sample()
        workflow.edges[u, v]["weight"] = workflow.edges[u, v]["weight_estimate"]
    
    network_speed = RandomVariable(samples=np.clip(np.random.normal(1, 0.3, 100), 1e-9, np.inf))
    network: nx.Graph = get_networks(
        num=1,
        cloud_name="chameleon",
        network_speed=network_speed,
        rv_weights=True
    )[0]

    for node in network.nodes:
        weight_rv: RandomVariable = network.nodes[node]["weight"]
        network.nodes[node]["weight_rv"] = weight_rv
        network.nodes[node]["weight_estimate"] = weight_rv.mean()
        network.nodes[node]["weight_actual"] = weight_rv.sample()
        network.nodes[node]["weight"] = network.nodes[node]["weight_estimate"]

    for (u, v) in network.edges:
        weight_rv: RandomVariable = network.edges[u, v]["weight"]
        network.edges[u, v]["weight_rv"] = weight_rv
        network.edges[u, v]["weight_estimate"] = weight_rv.mean()
        network.edges[u, v]["weight_actual"] = weight_rv.sample()
        network.edges[u, v]["weight"] = network.edges[u, v]["weight_estimate"]

    avg_task_cost = np.mean([workflow.nodes[node]["weight_actual"] for node in workflow.nodes])
    avg_dep_cost = np.mean([workflow.edges[u, v]["weight_actual"] for u, v in workflow.edges])
    avg_node_speed = np.mean([network.nodes[node]["weight_actual"] for node in network.nodes])
    avg_comm_speed = np.mean([network.edges[u, v]["weight_actual"] for u, v in network.edges])

    # ccr = (avg_task_cost / avg_node_speed) / (avg_dep_cost / avg_comm_speed)
    avg_node_speed_new = (avg_task_cost / ccr) / (avg_dep_cost / avg_comm_speed)
    avg_node_speed_multiplier = avg_node_speed_new / avg_comm_speed
    for node in network.nodes:
        network.nodes[node]["weight_rv"] *= avg_node_speed_multiplier
        network.nodes[node]["weight_estimate"] *= avg_node_speed_multiplier
        network.nodes[node]["weight_actual"] *= avg_node_speed_multiplier
        network.nodes[node]["weight"] = network.nodes[node]["weight_estimate"]

    return network, workflow

def get_offline_instance(network: nx.Graph, task_graph: nx.DiGraph) -> Tuple[nx.Graph, nx.DiGraph]:
    network = deepcopy(network)
    task_graph = deepcopy(task_graph)
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

    scheduler = HeftScheduler()
    scheduler_online = OnlineHeftScheduler()
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
    levels_range = [1, 2, 3, 4, 5]
    branching_range = [1, 2, 3, 4, 5]
    all_params = list(product(ccrs, levels_range, branching_range, range(n_samples)))

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
    df.to_csv("makespan_experiments.csv", index=False)

def analyze_results():
    df = pd.read_csv(thisdir / "makespan_experiments.csv")
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
    levels = 3
    branching_factor = 2
    ccr = 1
    sample_index = 0
    
    scheduler = HeftScheduler()
    scheduler_online = OnlineHeftScheduler()

    network, task_graph = get_instance(levels, branching_factor)

    schedule_online_naive = scheduler.schedule(network, task_graph)
    schedule_online_naive_actual = schedule_estimate_to_actual(network, task_graph, schedule_online_naive)
    makespan_online_naive = max(task.end for node_tasks in schedule_online_naive_actual.values() for task in node_tasks)

    schedule_online = scheduler_online.schedule(network, task_graph)
    makespan_online = max(task.end for node_tasks in schedule_online.values() for task in node_tasks)

    network_offline, task_graph_offline = get_offline_instance(network, task_graph)
    schedule_offline = scheduler.schedule(network_offline, task_graph_offline)
    makespan_offline = max(task.end for node_tasks in schedule_offline.values() for task in node_tasks)

    # Plot
    plotdir = thisdir / "example"
    plotdir.mkdir(exist_ok=True)

    ax: plt.Axes = draw_gantt(schedule=schedule_online_naive)
    ax.figure.savefig(plotdir / "naive_online_heft_schedule.png")
    ax: plt.Axes = draw_gantt(schedule=schedule_online_naive_actual)
    ax.figure.savefig(plotdir / "naive_online_heft_schedule_actual.png")
    ax: plt.Axes = draw_gantt(schedule=schedule_online)
    ax.figure.savefig(plotdir / "online_heft_schedule.png")
    ax: plt.Axes = draw_gantt(schedule=schedule_offline)
    ax.figure.savefig(plotdir / "offline_heft_schedule.png")

    ax: plt.Axes = draw_network(network)
    ax.figure.savefig(plotdir / "network_graph.png")
    ax: plt.Axes = draw_task_graph(task_graph)
    ax.figure.savefig(plotdir / "task_graph.png")




def main():
    # #record start time
    # start_time = time.perf_counter()

    # run_example()
    # run_experiment()
    # analyze_results()

    # #record end time
    # end_time = time.perf_counter()
    # elapsed = end_time - start_time
    # print(f"Execution time: {elapsed:.2f} seconds")

    
    workflow = get_wfcommons_instance(recipe_name="montage", ccr=1)

if __name__ == "__main__":
    main()