import logging
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np

from saga.scheduler import Task


def check_instance_simple(network: nx.Graph, task_graph: nx.DiGraph) -> None:
    """Check if the instance is valid.
    
    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Raises:
        ValueError: If the instance is invalid.
    """
    for node in network.nodes:
        if not isinstance(network.nodes[node]["weight"], (int, float)):
            raise ValueError(f"Node {node} has non-numeric weight (type({network.nodes[node]['weight']}).")
        if network.nodes[node]["weight"] <= 0 or 1/network.nodes[node]["weight"] <= 0:
            raise ValueError(f"Node {node} has zero, negative, or infinite weight.")
    for edge in network.edges:
        if not isinstance(network.edges[edge]["weight"], (int, float)):
            raise ValueError(f"Edge {edge} has non-numeric weight (type({network.edges[edge]['weight']}).")
        if network.edges[edge]["weight"] <= 0 or 1/network.edges[edge]["weight"] <= 0:
            raise ValueError(f"Edge {edge} has zero, negative, or infinite weight.")
    for node in task_graph.nodes:
        if not isinstance(task_graph.nodes[node]["weight"], (int, float)):
            raise ValueError(f"Node {node} has non-numeric weight (type({task_graph.nodes[node]['weight']}).")
        if task_graph.nodes[node]["weight"] <= 0 or 1/task_graph.nodes[node]["weight"] <= 0:
            raise ValueError(f"Node {node} has zero, negative, or infinite weight.")
    for edge in task_graph.edges:
        if not isinstance(task_graph.edges[edge]["weight"], (int, float)):
            raise ValueError(f"Edge {edge} has non-numeric weight (type({task_graph.edges[edge]['weight']}).")
        if task_graph.edges[edge]["weight"] <= 0 or 1/task_graph.edges[edge]["weight"] <= 0:
            raise ValueError(f"Edge {edge} has zero, negative, or infinite weight.")
    
    # check that network is fully connected
    if not nx.is_connected(network):
        raise ValueError("Network is not fully connected.")
    
    # check that task graph is acyclic
    if not nx.is_directed_acyclic_graph(task_graph):
        raise ValueError("Task graph is not acyclic.")
    
    # check that there is a single source and a single sink
    sources = [node for node in task_graph.nodes if task_graph.in_degree(node) == 0]
    if len(sources) != 1:
        raise ValueError("Task graph does not have exactly one source.")
    
    sinks = [node for node in task_graph.nodes if task_graph.out_degree(node) == 0]
    if len(sinks) != 1:
        raise ValueError("Task graph does not have exactly one sink.")
    

def get_insert_loc(schedule: List[Task], 
                   min_start_time: float, 
                   exec_time: float) -> Tuple[int, float]:
    """Get the location where the task should be inserted in the list of tasks.
    
    Args:
        schedule (List[Task]): The list of scheduled tasks.
        min_start_time (float): The minimum start time of the task.
        exec_time (float): The execution time of the task.
        
    Returns:
        Tuple[int, float]: The index and start time of the task.
    """
    if not schedule or min_start_time + exec_time < schedule[0].start:
        return 0, min_start_time
    
    for i, (left, right) in enumerate(zip(schedule, schedule[1:]), start=1):
        if min_start_time >= left.end and min_start_time + exec_time <= right.start:
            return i, min_start_time
        elif min_start_time < left.end and left.end + exec_time <= right.start:
            return i, left.end

    return len(schedule), max(min_start_time, schedule[-1].end)


class InvalidScheduleError(Exception):
    """Raised when a schedule is invalid."""
    def __init__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: Dict[Hashable, List[Task]],
                 message: str = "Invalid schedule.") -> None:
        self.network = network
        self.task_graph = task_graph
        self.schedule = schedule
        self.message = message

def standardize_task_graph(task_graph: nx.DiGraph) -> nx.DiGraph:
    """Standardize a task graph.

    Args:
        task_graph (nx.DiGraph): The task graph.

    Returns:
        nx.DiGraph: The standardized task graph.
    """
    task_graph = task_graph.copy()
    # add source and sink
    source = "__saga_src__"
    sink = "__saga_sink__"
    task_graph.add_node(source, weight=1e-9)
    task_graph.add_node(sink, weight=1e-9)
    for node in task_graph.nodes:
        if node in (source, sink):
            continue
        if task_graph.in_degree(node) == 0:
            task_graph.add_edge(source, node, weight=1e-9)
        if task_graph.out_degree(node) == 0:
            task_graph.add_edge(node, sink, weight=1e-9)

        task_graph.nodes[node]["weight"] = np.clip(task_graph.nodes[node]["weight"], 1e-9, 1e9)

    for edge in task_graph.edges:
        task_graph.edges[edge]["weight"] = np.clip(task_graph.edges[edge]["weight"], 1e-9, 1e9)


    return task_graph

def standardize_network(network: nx.Graph) -> nx.Graph:
    """Standardize a network graph.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        nx.Graph: The standardized network graph.
    """
    network = network.copy()
    for node in network.nodes:
        network.nodes[node]["weight"] = np.clip(network.nodes[node]["weight"], 1e-9, 1e9)
    for edge in network.edges:
        network.edges[edge]["weight"] = np.clip(network.edges[edge]["weight"], 1e-9, 1e9)

    return network

def validate_simple_schedule(network: nx.Graph, task_graph: nx.DiGraph, schedule: Dict[Hashable, List[Task]]) -> None:
    """Validate a simple schedule.

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.
        schedule (Dict[Hashable, List[Task]]): The schedule.

    Raises:
        ValueError: If instance is invalid.
        InvalidScheduleError: If schedule is invalid.
    """
    check_instance_simple(network, task_graph) # check that instance is valid

    tasks = {task.name: task for node in schedule for task in schedule[node]}
    # check that all tasks are scheduled
    if len(tasks) != len(task_graph.nodes):
        logging.error(f"Only {len(tasks)} tasks are scheduled.")
        raise InvalidScheduleError(network, task_graph, schedule)

    # check that schedule is valid
    for node in schedule:
        if node not in network.nodes:
            logging.error(f"Node {node} is not in the network.")
            raise InvalidScheduleError(network, task_graph, schedule)

        # check that tasks on node do not overlap
        for left, right in zip(schedule[node], schedule[node][1:]):
            if not (np.isclose(left.end, right.start) or left.end < right.start):
                message = f"Tasks {left} and {right} overlap on node {node}."
                raise InvalidScheduleError(network, task_graph, schedule, message)

        for task in schedule[node]:
            # check that task runtime is correct
            if not np.isclose(task_graph.nodes[task.name]["weight"] / network.nodes[node]["weight"], task.end - task.start):
                message = f"Task {task} has incorrect runtime: {task.end - task.start}. Expected {task_graph.nodes[task.name]['weight'] / network.nodes[node]['weight']}."
                raise InvalidScheduleError(network, task_graph, schedule, message)

            # check that task start time is feasible
            for parent in task_graph.predecessors(task.name):
                parent_node = tasks[parent].node
                parent_data_arrival_time = task_graph.edges[parent, task.name]["weight"] / network.edges[parent_node, node]["weight"] + tasks[parent].end
                if not (np.isclose(parent_data_arrival_time, task.start) or parent_data_arrival_time < task.start):
                    message = f"Task {task} has incorrect start time: {task.start}. Expected {parent_data_arrival_time}."
                    raise InvalidScheduleError(network, task_graph, schedule, message)
                
            # check that task end time is correct
            if not np.isclose(task.start + task_graph.nodes[task.name]["weight"] / network.nodes[node]["weight"], task.end):
                logging.error(f"Task {task} has incorrect end time: {task.end}. Expected {task.start + task_graph.nodes[task.name]['weight'] / network.nodes[node]['weight']}.")
                raise InvalidScheduleError(network, task_graph, schedule, message)

