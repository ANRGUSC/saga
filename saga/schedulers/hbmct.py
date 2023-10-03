import logging  
import pathlib
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np

from ..scheduler import Scheduler, Task
from ..utils.tools import get_insert_loc
from .heft import heft_rank_sort

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

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        List[Hashable]: The sorted list of tasks.
    """
    rankings = hbmct_rank_sort(network, task_graph)
    groups = [[rankings[0]]]
    for task_name in rankings[1:]:
        is_same_group = True
        for predecessors in task_graph.predecessors(task_name):
            if predecessors in groups[-1]:
                groups.append([task_name])
                is_same_group = False
                break
        if is_same_group:
            groups[-1].append(task_name)
    print(groups)
    return groups

def calculate_est(network: nx.Graph,
                  task_graph: nx.DiGraph,
                  group:List[Hashable],
                  commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
                  comp_schedule: Dict[Hashable, List[Task]],
                  task_schedule: List[Hashable]) -> Dict[Hashable, Dict[Hashable, float]]:
    
    est_table = {task_name:{node:None for node in network.nodes} for task_name in group}
    for task_name in group:
        for node in comp_schedule:
            max_arrival_time: float = max( 
                    [
                        0.0, *[
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
                            group:List[Hashable],
                            est_table:Dict[Hashable, Dict[Hashable, float]]
                            ) -> Dict[Hashable, List[Hashable]]:
    assignments = {node:[] for node in network.nodes}
    for task_name in group: #Assign nodes based on execution times
        assigned_node = min(runtimes[task_name], key= runtimes[task_name].get)
        assignments[assigned_node].append(task_name)
    for node in assignments: #sort based on earliest start times
        assignments[node].sort(key= lambda x, node=node: est_table[x][node])
    print("assignments", assignments)
    return assignments
    
def get_ft(node_schedule: List[Task]) -> float:
    if node_schedule:
        return node_schedule[-1].end
    return 0

def get_ft_after_insert(new_task_name,
                        node,
                        assignments: List[Hashable],
                        node_schedule: List[Task],
                        est_table,
                        runtimes,
                        insert_position:int) -> (float, List[Task], Task):
    new_assignments = assignments.copy()
    new_assignments.append(new_task_name) #Todo: O(n) time insertion insted of sorting
    new_assignments.sort(key = lambda task: est_table[new_task_name][node])
    new_schedule = node_schedule.copy()
    new_task = None
    start_time = 0
    if new_schedule:
        start_time = max(start_time, node_schedule[-1].end)
    for task_id in reversed(range(insert_position, len(new_schedule))):
        del new_schedule[task_id]
    for task_name in new_assignments:
        #Todo: Calculate where to start inserting new_assignments from delete everything before
        start_time = est_table[task_name][node]
        if new_schedule:
            start_time = max(start_time, new_schedule[-1].end)
        # print("start_time", start_time, "runtime", runtimes[task_name][node])
        task = Task(node, task_name, start_time, start_time + runtimes[task_name][node])
        new_schedule.append(task)
        if task_name == new_task_name:
            new_task = task
    return new_schedule[-1].end, new_schedule, new_task

def delete_task_from_schedule(task_name, node, node_schedule: List[Task], est_table, runtimes):
    new_schedule = node_schedule.copy()
    for i in range(len(new_schedule)):
        if new_schedule[i].name == task_name:
            del new_schedule[i]
            break

    for j in range(i, len(new_schedule)):
        task = new_schedule[j]
        start_time = est_table[task.name][node]
        if j!=0:
            start_time = max(start_time, new_schedule[j].end)
            new_schedule[j] = Task(node, task_name, start_time, start_time + runtimes[task_name][node])
    new_ft = 0
    if new_schedule:
        new_ft = new_schedule[-1].end
    return new_ft, new_schedule

class HbmctScheduler(Scheduler):
    """Schedules tasks using the HBMCT algorithm."""
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
            for task in task_graph.nodes:
                cost: float = task_graph.nodes[task]["weight"]
                runtimes[node][task] = cost / speed
                logging.debug("Task %s on node %s has runtime %s", task, node, runtimes[node][task])

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
                  commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]):
        """
        The BMCT Heuristic
        """
        comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}
        for group in groups:
            comp_group_start_positions = {node: len(comp_schedule[node]) for node in comp_schedule}
            est_table = calculate_est(network, task_graph, group, commtimes, comp_schedule, task_schedule)
            print("EST:",est_table)
            average_est = {task_name: np.mean([
                est for est in est_table[task_name]
            ]) for task_name in est_table}
            assignments = get_initial_assignments(network, runtimes, group, est_table)
            for node in assignments:
                for task_name in assignments[node]:
                    start_time = est_table[task_name][node]
                    if comp_schedule[node]:
                        start_time = max(start_time, comp_schedule[node][-1].end)
                    task = Task(node, task_name, start_time, start_time + runtimes[task_name][node])
                    comp_schedule[node].append(task)
                    task_schedule[task_name] = task
            print("Schedule BEFORE BMCT:", comp_schedule)
            assignment_changed = True
            while assignment_changed:
                assignment_changed = False
                max_ft_node = max(comp_schedule, key = lambda x: get_ft(comp_schedule[x]))
                max_ft = comp_schedule[max_ft_node][-1].end
                print(f"current MFT {max_ft_node} with finish time {max_ft}")
                
                avg_est_assignments = sorted(assignments[max_ft_node], key=lambda task_name, average_est = average_est: average_est[task_name])
                
                for task_name in avg_est_assignments:
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
                                # print(f"New finish time for node {node} after inserting task {task_name}: {new_ft}")
                                if new_ft<max_ft and new_ft<min_ft:
                                    min_ft = new_ft
                                    min_ft_node = node
                                    min_ft_node_schedule = node_schedule.copy()
                                    updated_task = new_task
                                    # print("Found better!")
                        if min_ft_node:
                            #Update the schedules
                            assignment_changed=True
                            comp_schedule[min_ft_node] = min_ft_node_schedule
                            comp_schedule[max_ft_node] = max_ft_node_new_schedule
                            task_schedule[task_name] = updated_task
                            print(f"New finish time for node {min_ft_node} after inserting task {task_name}: {new_ft}")
                            print("Schedule AFTER BMCT:", comp_schedule)
                            break


            print("***********************************")
                # for task_name in avg_est_assignments:




        
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[str, List[Task]]:
        # print("scheduling...")
        runtimes, commtimes = HbmctScheduler.get_runtimes(network, task_graph)
        print("Runtimes:", runtimes)
        groups = hbmct_create_groups(network, task_graph)
        self.schedule_groups(network, task_graph, groups, runtimes, commtimes)
        return None