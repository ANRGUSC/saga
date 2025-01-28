from saga.schedulers import HeftScheduler
from saga.scheduler import Scheduler, Task
# from saga.utils.random_variable import RandomVariable

from typing import Dict, Hashable, List
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
        task = Task( #creating a "new" task with all of our calculated attributes 
            node=task_node,
            name=task_name,
            start=start_time,
            end=start_time + runtime
        )
        schedule_actual[task_node].append(task) #putting into our actual scheduler 
        tasks_actual[task_name] = task #putting into our dict of actual tasks 

    return schedule_actual


class OnlineHeftScheduler(Scheduler):
    def __init__(self):
        self.heft_scheduler = HeftScheduler()

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """
        A 'live' scheduling loop using HEFT in a dynamic manner.
        We assume each node/task has 'weight_actual' and 'weight_estimate' attributes.
        
        1) Updates node, edge, and task weights to use 'actual' or 'estimate' 
            based on whether a task is in schedule_actual.
        2) Calls the standard HEFT for an 'estimated' schedule.
        3) Converts that schedule to 'actual' times using schedule_estimate_to_actual.
        4) Commits the earliest-finishing new task to schedule_actual.
        5) Repeats until all tasks are scheduled.        

        comp_schedule = {
        "Node1": [Task(node="Node1", name="TaskA", start=0, end=5)],
        "Node2": [Task(node="Node2", name="TaskB", start=3, end=8)],
        """

        #final actual schedule dict
        schedule_actual: Dict[Hashable, List[Task]] = {
            node: [] for node in network.nodes
        }

        # Total number of tasks in the graph
        total_tasks = len(task_graph.nodes)

        # Helper function: See if a task is already in schedule_actual
        def is_task_committed(task_name: str) -> bool:
            for node_tasks in schedule_actual.values():
                for t in node_tasks:
                    if t.name == task_name:
                        return True
            return False

        while True:



            '''
            for task_name in task_graph.nodes:
                if is_task_committed(task_name):
                    # Task is already in final schedule => use actual
                    task_graph.nodes[task_name]["weight"] = task_graph.nodes[task_name]["weight_actual"]
                else:
                    # Otherwise => estimate
                    task_graph.nodes[task_name]["weight"] = task_graph[task_name]["weight_estimate"]

            for (u, v) in task_graph.edges:
                if is_task_committed(u) and is_task_committed(v):
                    # Task is already in final schedule => use actual
                    task_graph.edges[u, v]["weight"] = task_graph.edges[u, v]["weight_actual"]
                else:
                    # Otherwise => estimate
                    task_graph.edges[u, v]["weight"] = task_graph.edges[u, v]["weight_estimate"]

            for node in network.nodes:
                # if the node is in schedule_actual Assume speed of node is known? we should test for varying speed of node?
                if schedule_actual[node]:
                    network.nodes[node]["weight"] = network.nodes[node]["weight_actual"]
                else:
                    network.nodes[node]["weight"] = network.nodes[node]["weight_estimate"]

            for (src, dst) in network.edges:
                """
                edges = [(1, 2), (1, 3), (2, 3)] example of what they look like
                """
                if schedule_actual[src] and schedule_actual[dst]:
                    #if both nodes are in actual schedule then get actual
                    network.edges[src, dst]["weight"] = network.edges[src, dst]["weight_actual"]
                else:
                    # Otherwise => estimate
                    network.nodes[node]["weight"] = network.nodes[node]["weight_estimate"]
            '''

            
            #------------------------------------------Step 2: Get next task to finish based on *actual* execution time
            
            schedule_estimate = self.heft_scheduler.schedule(network, task_graph, schedule_actual)
            schedule_actual_hypothetical = schedule_estimate_to_actual(network, task_graph, schedule_estimate)

            all_tasks_hypothetical = []
            # create list of dict values
            for node_tasks in schedule_actual_hypothetical.values():
                all_tasks_hypothetical.extend(node_tasks)

            estimated_tasks_only = []

            for task in all_tasks_hypothetical:
                # check if task is not already been committed before
                if not is_task_committed(task.name):
                    # add the task to the estimated_tasks_only list
                    estimated_tasks_only.append(task)
 
            if not estimated_tasks_only:
                break
            else:
                # sort by lowest end time
                estimated_tasks_only.sort(key=lambda x: x.end)
                next_task_to_commit = estimated_tasks_only[0]
                #add the lowest finishing task to the actual schedule
                schedule_actual[next_task_to_commit.node].append(next_task_to_commit)

            #------------------------------------------Step 3: check if all tasks have been added to schedule actual
            committed_count = 0

            #loop through each node, schedule_actual.values() returns a list of tasks on that node 
            for tasks in schedule_actual.values():
                committed_count += len(tasks)

            if committed_count >= total_tasks:
                break

        return schedule_actual