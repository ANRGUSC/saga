import logging
from typing import Dict, Hashable, List, Tuple, Optional

import networkx as nx
import numpy as np

from copy import deepcopy

from saga.scheduler import Task


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
    # print(f"Task order: {[(task, tasks_estimate[task].start, tasks_estimate[task].end) for task in task_order]}") #debugging print to see the order of tasks
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

        #If current node already has tasks scheduled, we set the start time to be the end of the last task on that node
        if schedule_actual[task_node]: 
            start_time = max(start_time, schedule_actual[task_node][-1].end)

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


#maybe move this to a different file in utils 
class ScheduleInjector:
    def schedule(
        self,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        schedule: Optional[Dict[Hashable, List[Task]]] = None,
        min_start_time: float = 0.0,
        **algo_kwargs
    ) -> Dict[Hashable, List[Task]]:
        # 1. Initialize or clone
        if schedule is None:
            comp_schedule: Dict[Hashable, List[Task]] = {
                node: [] for node in network.nodes
            }
            task_map: Dict[Hashable, Task] = {}
            current_moment = min_start_time
            
        else:
            comp_schedule = deepcopy(schedule)
            # Flatten out all Task objects for quick look-ups
            task_map = {
                t.name: t
                for node_tasks in comp_schedule.values()
                for t in node_tasks
            }
            # Advance clock to the latest end-time seen
            current_moment = min(
                (t.end for t in task_map.values() if t.end > min_start_time),
                default=min_start_time
            )
            

        # 2. Delegate to algorithm core
        #    Pass along comp_schedule, task_map, current_moment, plus any extra args
        updated_schedule = self._do_schedule(
            network,
            task_graph,
            comp_schedule,
            task_map,
            current_moment,
            **algo_kwargs
        )

        # 3. Return the mutated schedule
        return updated_schedule