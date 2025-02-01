from saga.schedulers import HeftScheduler
from saga.scheduler import Scheduler, Task
# from saga.utils.random_variable import RandomVariable

import pygraphviz as pgv
from typing import Dict, Hashable, List, Tuple
import networkx as nx


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

        print("Final schedule:", schedule_actual)
        return schedule_actual
    
from saga.utils.random_graphs import get_branching_dag, get_network, get_diamond_dag, get_chain_dag, get_fork_dag
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_instance() -> Tuple[nx.Graph, nx.DiGraph]:
    # Create a random network
    network = get_network(num_nodes=4)
    # Create a random task graph
    #task_graph = get_fork_dag()
    task_graph = get_branching_dag(levels=3, branching_factor=5)
    #task_graph = get_chain_dag(num_nodes=10)
    #task_graph = get_diamond_dag()
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


def main():
    scheduler = HeftScheduler()
    scheduler_online = OnlineHeftScheduler()

    n_samples = 100

    rows = []
    worst_instance, worst_makespan_ratio = None, 0
    for _ in range(n_samples):
        network, task_graph = get_instance()
        schedule = scheduler.schedule(network, task_graph)
        schedule_actual = schedule_estimate_to_actual(network, task_graph, schedule)
        makespan = max(task.end for node_tasks in schedule_actual.values() for task in node_tasks)

        schedule_online = scheduler_online.schedule(network, task_graph)
        makespan_online = max(task.end for node_tasks in schedule_online.values() for task in node_tasks)

        # print(f"HEFT makespan: {makespan}")
        # print(f"Online HEFT makespan: {makespan_online}")

        if makespan_online / makespan > worst_makespan_ratio:
            worst_instance = (network, task_graph)
            worst_makespan_ratio = makespan_online / makespan


        rows.append((makespan, makespan_online, makespan_online / makespan))

    df = pd.DataFrame(rows, columns=["HEFT", "Online HEFT", "Makespan Ratio"])
    print(df.describe())

    fig, ax = plt.subplots()
    
    # boxplot
    df.boxplot(column=["Makespan Ratio"], ax=ax)
    ax.set_ylabel("Makespan")


    fig.savefig("online_heft.png")


    # Draw task graph, network, and schedules for the worst instance
    network, task_graph = worst_instance
    schedule = scheduler.schedule(network, task_graph)
    schedule_actual = schedule_estimate_to_actual(network, task_graph, schedule)
    makespan_actual = max(task.end for node_tasks in schedule_actual.values() for task in node_tasks)
    schedule_online = scheduler_online.schedule(network, task_graph, do_print=True)
    makespan_online = max(task.end for node_tasks in schedule_online.values() for task in node_tasks)
    xmax = max(makespan_actual, makespan_online)

    ax_task_graph: plt.Axes = draw_task_graph(task_graph)
    ax_task_graph.get_figure().savefig("task_graph.png")

    ax_network: plt.Axes = draw_network(network)
    ax_network.get_figure().savefig("network.png")

    ax_schedule: plt.Axes = draw_gantt(schedule_actual, xmax=xmax)
    ax_schedule.get_figure().savefig("schedule.png")

    ax_schedule_online: plt.Axes = draw_gantt(schedule_online, xmax=xmax)
    ax_schedule_online.get_figure().savefig("schedule_online.png")



if __name__ == "__main__":
    main()