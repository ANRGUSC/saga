import logging  
from typing import Dict, Hashable, List, Tuple, Optional

import networkx as nx
import numpy as np

from ....scheduler import Scheduler, Task
from ...heft import heft_rank_sort

def hbmct_rank_sort(network: nx.Graph, task_graph: nx.DiGraph) -> List[Hashable]:    
    """Sort tasks based on their rank (as defined in the HEFT paper).

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        List[Hashable]: The sorted list of tasks.
    """

    return heft_rank_sort(network, task_graph)

def hbmct_create_groups(network: nx.Graph, task_graph: nx.DiGraph) -> List[List[Hashable]]:
    """Create Independent Groups for scheduling. 
    Keep going down the sorted list and create a new group everytime there is a task depending on the current groups.

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        List[List[Hashable]]: List of groups.
    """
    rankings = hbmct_rank_sort(network, task_graph)
    groups = [[rankings[0]]]
    logging.debug("Upward: Rankings %s", rankings)
    for task_name in rankings[1:]:
        is_same_group = True
        for predecessor in task_graph.predecessors(task_name):
            if predecessor in groups[-1]:
                groups.append([task_name])
                is_same_group = False
                break
        if is_same_group:
            groups[-1].append(task_name)
    return groups

def calculate_est(network: nx.Graph,
                  task_graph: nx.DiGraph,
                  group:List[Hashable],
                  commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
                  comp_schedule: Dict[Hashable, List[Task]],
                  task_schedule: Dict[Hashable, Task],
                  task_graph_arrival_time) -> Dict[Hashable, Dict[Hashable, float]]:
    """
    Calculate the earliest start times for the given group on all nodes. 

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.
        group (List[Hashable]): The independent group of tasks
        commtimes (Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]): Communication times for node-pair task-pair
        comp_schedule (Dict[Hashable, List[Task]]): schedule of tasks for each node
        task_schedule Dict[Hashable, Task]): Task objects associated with each task
    Returns:
        Dict[Hashable, Dict[Hashable, float]]: Earliest Start Time table for the given group
    """
    est_table = {task_name:{node:None for node in network.nodes} for task_name in group}
    for task_name in group:
        for node in comp_schedule:
            max_arrival_time: float = max(
                    [
                        task_graph_arrival_time, *[
                            task_schedule[parent].end + (
                                commtimes[(task_schedule[parent].node, node)][(parent, task_name)]
                            )
                            for parent in task_graph.predecessors(task_name)
                        ]
                    ]
                )
            if comp_schedule[node]:
                est_table[task_name][node] = max(max_arrival_time, comp_schedule[node][-1].end)
            else:
                est_table[task_name][node] = max_arrival_time
    return est_table
  
def get_initial_assignments(network: nx.Graph,
                            runtimes: Dict[Hashable, Dict[Hashable, float]],
                            group: List[Hashable],
                            est_table: Dict[Hashable, Dict[Hashable, float]]
                            ) -> Dict[Hashable, List[Hashable]]:
    """
    Get initial assignments of the tasks of an independent groups based on their execution times

    Args:
        network (nx.Graph): The node network 
        runtimes (Dict[Hashable, Dict[Hashable, float]]): Runtimes of each node for given tasks
        group (List[Hashable]): Current independent group under consideration
        est_table (Dict[Hashable, Dict[Hashable, float]]): Earliest Start Time table for the given group
    
    Returns:
        Dict[Hashable, List[Hashable]]: Initial Schedule of tasks

    """
    assignments = {node:[] for node in network.nodes}
    for task_name in group: #Assign nodes based on execution times
        assigned_node = min(runtimes, key= lambda node, task_name=task_name : runtimes[node][task_name])
        assignments[assigned_node].append(task_name)
    for node in assignments: #sort based on earliest start times
        assignments[node].sort(key= lambda x, node=node: est_table[x][node])
    return assignments
    
def get_ft(node_schedule: List[Task]) -> float:
    """
    Calculate finish time for a node in a given schedule

    Args:
        node_schedule (List[Task]): Schedule of the node

    Returns:
        float: Finish Time of the node 
    """
    if node_schedule:
        return node_schedule[-1].end
    return 0

def get_ft_after_insert(new_task_name: Hashable,
                        node: Hashable,
                        assignments: List[Hashable],
                        node_schedule: List[Task],
                        est_table: Dict[Hashable, Dict[Hashable, float]],
                        runtimes: Dict[Hashable, Dict[Hashable, float]],
                        insert_position:int) -> (float, List[Task], Task):
    """
    Calculate the finish time after inserting task in the schedule of a node.

    Args:
        new_task_name (Hashable) New task to be inserted.
        node (nx.Graph): Node on which the task is to be run
        assignments (List[Hashable]): Assignments of the tasks
        node_schedule (List[Task]): Schedule of the node.
        est_table (Dict[Hashable, Dict[Hashable, float]]): Earliest Start Time table for the given group
        runtimes (Dict[Hashable, Dict[Hashable, float]]): Runtimes of each node for given tasks
        insert_postiion (int): Position at which current group of tasks were inserted
    Returns:
        float: Finish time after insert
        List[Task]: New schedule after insertion
        Task: Task object of the task after inserting
    """
    new_assignments = assignments.copy()
    new_assignments.append(new_task_name) #Todo: O(n) time insertion instead of sorting
    new_assignments.sort(key = lambda task: est_table[new_task_name][node])
    new_schedule = node_schedule.copy()
    new_task = None
    start_time = 0
    if new_schedule:
        start_time = max(start_time, node_schedule[-1].end)
    for task_id in reversed(range(insert_position, len(new_schedule))):
        del new_schedule[task_id]
    for task_name in new_assignments:
        #Todo: Calculate where to start inserting new_assignments from instead of deleting
        start_time = est_table[task_name][node]
        if new_schedule:
            start_time = max(start_time, new_schedule[-1].end)
        task = Task(node, task_name, start_time, start_time + runtimes[node][task_name])
        new_schedule.append(task)
        if task_name == new_task_name:
            new_task = task
    return new_schedule[-1].end, new_schedule, new_task

def delete_task_from_schedule(task_name:Hashable,
                              node: Hashable,
                              node_schedule: List[Task],
                              est_table: Dict[Hashable, Dict[Hashable, float]],
                              runtimes: Dict[Hashable, Dict[Hashable, float]]
                              )-> (float, List[Task]):
    """
    Calculate the new schedule after removing task from a node schedule.

    Args:
        task_name (Hashable) New task to be inserted.
        node (nx.Graph): Node on which the task is to be run
        node_schedule (List[Task]): Schedule of the node.
        est_table (Dict[Hashable, Dict[Hashable, float]]): Earliest Start Time table for the given group
        runtimes (Dict[Hashable, Dict[Hashable, float]]): Runtimes of each node for given tasks
    Returns:
        float: Finish time after deletion
        List[Task]: New schedule after insertion
    """
    new_schedule = node_schedule.copy()
    i = None
    for i, task in enumerate(new_schedule):
        if task.name == task_name:
            del new_schedule[i]
            logging.debug("Deleted: %s", new_schedule)
            break
    for j in range(i, len(new_schedule)):
        task = new_schedule[j]
        start_time = est_table[task.name][node]
        if j!=0:
            start_time = max(start_time, new_schedule[j-1].end)
        new_schedule[j] = Task(node, task.name, start_time, start_time + runtimes[node][task.name])

    new_ft = 0
    logging.debug("Schedule of %s after remoiving %s : %s",node, task_name, new_schedule)
    if new_schedule:
        new_ft = new_schedule[-1].end
    return new_ft, new_schedule

class ResidualHbmctScheduler(Scheduler):
    """Schedules tasks using the HBMCT (Hybrid Minimum Completion Time) algorithm.

    Source: https://dx.doi.org/10.1137/0218016
    """
    @staticmethod
    def get_runtimes(network: nx.Graph,
                     task_graph: nx.DiGraph) -> Tuple[Dict[Hashable, Dict[Hashable, float]],
                                                      Dict[Tuple[Hashable, Hashable],
                                                           Dict[Tuple[Hashable, Hashable], float]]]:
        """Get the expected runtimes of all tasks on all nodes.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Tuple[Dict[Hashable, Dict[Hashable, float]],
                  Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]]:
                A tuple of dictionaries mapping nodes to a dictionary of tasks and their runtimes
                and edges to a dictionary of tasks and their communication times. The first dictionary
                maps nodes to a dictionary of tasks and their runtimes. The second dictionary maps edges
                to a dictionary of task dependencies and their communication times.
        """
        runtimes = {}
        for node in network.nodes:
            runtimes[node] = {}
            speed: float = network.nodes[node]["weight"]
            for task_name in task_graph.nodes:
                cost: float = task_graph.nodes[task_name]["weight"]
                runtimes[node][task_name] = cost / speed
                logging.debug("Task %s on node %s has runtime %s", task_name, node, runtimes[node][task_name])

        commtimes = {}
        for src, dst in network.edges:
            commtimes[src, dst] = {}
            commtimes[dst, src] = {}
            speed: float = network.edges[src, dst]["weight"]
            for src_task, dst_task in task_graph.edges:
                cost = task_graph.edges[src_task, dst_task]["weight"]
                commtimes[src, dst][src_task, dst_task] = cost / speed
                commtimes[dst, src][src_task, dst_task] = cost / speed
                logging.debug(
                    "Task %s on node %s to task %s on node %s has communication time %s",
                    src_task, src, dst_task, dst, commtimes[src, dst][src_task, dst_task]
                )

        return runtimes, commtimes
    
    @staticmethod
    def schedule_groups(network: nx.Graph,
                  task_graph: nx.DiGraph,
                  groups:List[List[Hashable]],
                  runtimes: Dict[Hashable, Dict[Hashable, float]],
                  commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
                  current_schedule: Optional[Dict[str, List[Task]]] = None,
                  task_graph_arrival_time: float = 0.0
                  ) -> Dict[Hashable, List[Task]]:
        """
        Schedule all the groups independently

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            groups (List[List[Hashable]]): List of groups
            runtimes (Dict[Hashable, Dict[Hashable, float]]): Runtimes of each node for given tasks
            commtimes (Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]): Communication times for node-pair task-pair
        
        Returns:
            Dict[Hashable, List[Task]]: The schedule for each node
        """
        comp_schedule: Dict[Hashable, List[Task]] = current_schedule or {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}
        for group in groups:
            comp_group_start_positions = {node: len(comp_schedule[node]) for node in comp_schedule}
            est_table = calculate_est(network, task_graph, group, commtimes, comp_schedule, task_schedule, task_graph_arrival_time)
            average_est = {task_name: np.mean([
                est for est in est_table[task_name]
            ]) for task_name in est_table}
            assignments = get_initial_assignments(network, runtimes, group, est_table)
            for node in assignments:
                for task_name in assignments[node]:
                    start_time = est_table[task_name][node]
                    if comp_schedule[node]:
                        start_time = max(start_time, comp_schedule[node][-1].end)
                    task = Task(node, task_name, start_time, start_time + runtimes[node][task_name])
                    comp_schedule[node].append(task)
                    task_schedule[task_name] = task
            logging.debug("Initial assignment for group %s: %s", group, comp_schedule)
            assignment_changed = True
            while assignment_changed:
                assignment_changed = False
                max_ft_node = max(comp_schedule, key = lambda x: get_ft(comp_schedule[x]))
                max_ft = comp_schedule[max_ft_node][-1].end
                avg_est_assignments = sorted(assignments[max_ft_node], key=lambda task_name, average_est = average_est: average_est[task_name])
                logging.debug("current MFT %s with finish time %s", max_ft_node, max_ft)
                
                for task_name in avg_est_assignments:
                    logging.debug("Trying to move around task %s", task_name)
                    new_max_ft, max_ft_node_new_schedule = delete_task_from_schedule(
                        task_name,
                        max_ft_node,
                        comp_schedule[max_ft_node],
                        est_table,
                        runtimes
                    )
                    if new_max_ft<max_ft: #Check if removing the task actually improves the max_ft for the node
                        min_ft_node = None
                        min_ft_node_schedule = None
                        updated_task = None
                        min_ft = float("inf")
                        for node in network.nodes:
                            if node!=max_ft_node:
                                new_ft, node_schedule, new_task = get_ft_after_insert(task_name,
                                                            node,
                                                            assignments[node],
                                                            comp_schedule[node],
                                                            est_table,
                                                            runtimes,
                                                            comp_group_start_positions[node])
                                logging.debug("EST of task %s on node %s: %s", task_name, node, est_table[task_name][node])
                                logging.debug("New finish time for node %s after inserting task %s: %s", node, task_name, new_ft)
                                logging.debug("Schedule of node %s after inserting %s : %s",node, task_name, node_schedule)
                                if new_ft<max_ft and new_ft<min_ft:
                                    min_ft = new_ft
                                    min_ft_node = node
                                    min_ft_node_schedule = node_schedule.copy()
                                    updated_task = new_task
                        if min_ft_node:
                            assignment_changed=True
                            comp_schedule[min_ft_node] = min_ft_node_schedule
                            comp_schedule[max_ft_node] = max_ft_node_new_schedule
                            task_schedule[task_name] = updated_task
                            assignments[min_ft_node].append(task_name)
                            assignments[max_ft_node].remove(task_name)
                            logging.debug("New assignment for group %s: %s", group, comp_schedule)

                            break
        return comp_schedule



        
    def schedule(self, network: nx.Graph, task_graphs: List[Tuple[nx.DiGraph, float]]) -> Dict[str, List[Task]]:
        """Computes the schedule for the task graph using the CPoP algorithm.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[str, List[Task]]: The schedule for the task graph.
        """
        comp_schedule: Dict[Hashable, List[Task]] = None
        # {node: [] for node in network.nodes}

        for task_graph_tupple in task_graphs:
            task_graph = task_graph_tupple[0]
            task_graph_arrival_time = task_graph_tupple[1]

            runtimes, commtimes = ResidualHbmctScheduler.get_runtimes(network, task_graph)
            groups = hbmct_create_groups(network, task_graph)
            comp_schedule = self.schedule_groups(network, task_graph, groups, runtimes, commtimes, comp_schedule, task_graph_arrival_time)

        return comp_schedule
