import logging
import math
from dataclasses import dataclass
import pprint
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from saga.utils.random_variable import RandomVariable


@dataclass
class Task:
    node: str
    name: str
    start: Optional[float]
    end: Optional[float] 

def format_dag(dag: nx.DiGraph) -> str:
    """Formats a DAG in a readable format

    Args:
        dag: DAG
    
    Returns:
        Formatted DAG
    """
    text = ""
    for node in dag.nodes:
        text += f"{node}: {dag.nodes[node]}\n"
    for edge in dag.edges:
        text += f"{edge[0]} -> {edge[1]}: {dag[edge[0]][edge[1]]}\n"
    return text

def format_network(network: nx.Graph) -> str:
    """Formats a network in a readable format

    returns table of pairwise edge weights

    Args:
        network: Network
    
    Returns:
        Formatted network
    """
    comp_speeds = {node: network.nodes[node]["weight"] for node in network.nodes}
    df = pd.DataFrame(
        [
            [
                network.edges[(node1, node2)]["weight"]
                for node2 in network.nodes
            ]
            for node1 in network.nodes
        ],
        index=network.nodes,
        columns=network.nodes
    )
    return pprint.pformat(comp_speeds) + "\n" + df.to_string()

def validate_schedule(network: nx.Graph, task_graph: nx.DiGraph, schedule: Dict[str, List[Task]]):
    """Validates a schedule

    Args:
        network: Network
        task_graph: Task graph

    Raises:
        ValueError: If the schedule is invalid
    """
    network, task_graph = standardize_instance(network, task_graph)

    tasks = {task.name: task for node, tasks in schedule.items() for task in tasks}
    schedule = {node: sorted(tasks, key=lambda task: task.end) for node, tasks in schedule.items()}
    for node, node_tasks in schedule.items():
        for i, task in enumerate(node_tasks):
            parents = list(task_graph.predecessors(task.name))
            parent_arrival_times = {
                parent: tasks[parent].end + (
                    task_graph.edges[(parent, task.name)]["weight"] /
                    network.edges[(tasks[parent].node, node)]["weight"]
                )
                for parent in parents
            }
            arrival_time = max(parent_arrival_times.values()) if len(parent_arrival_times) > 0 else 0
            if i > 0:
                arrival_time = max(arrival_time, node_tasks[i-1].end)

            duration = task_graph.nodes[task.name]["weight"] / network.nodes[node]["weight"]
            if not math.isclose(task.start, arrival_time, abs_tol=1e-6) or not math.isclose(task.end, task.start + duration, abs_tol=1e-6):
                logging.error(f"Task {task.name} should start at {arrival_time} but starts at {task.start}")
                logging.error(f"Task {task.name} should end at {arrival_time + duration} but ends at {task.end}")
                logging.error(f"Parents: {parent_arrival_times}")
                if i > 0:
                    logging.error(f"Previous task: {node_tasks[i-1]}")
                logging.error(f"Schedule: {schedule}")
                logging.error(f"Task graph:\n{format_dag(task_graph)}")
                logging.error(f"Network:\n{format_network(network)}")
                logging.error(f"Duration: {duration}")
                raise Exception("Invalid schedule")

SRC = "__src__"
DST = "__dst__"
def standardize_instance(network: nx.Graph, 
                         task_graph: nx.DiGraph) -> Tuple[nx.Graph, nx.DiGraph]:
    """Standardizes a problem instance

    Args:
        network: Network
        task_graph: Task graph

    Returns:
        Standardized network and task graph
    """
    task_graph = task_graph.copy()
    network = network.copy()

    if SRC in task_graph.nodes or DST in task_graph.nodes:
        logging.warning(f"Skipping standardization because {SRC} or {DST} already in task graph")
        return network, task_graph

    root_nodes = [node for node in task_graph.nodes if task_graph.in_degree(node) == 0]
    leaf_nodes = [node for node in task_graph.nodes if task_graph.out_degree(node) == 0]

    weights_are_random = False
    for node in task_graph.nodes:
        if isinstance(task_graph.nodes[node]["weight"], RandomVariable):
            weights_are_random = True
            break

    zero_weight = RandomVariable([1e-9]) if weights_are_random else 1e-9

    # add src and dst nodes to task_graph
    task_graph.add_node(SRC, weight=zero_weight)
    task_graph.add_node(DST, weight=zero_weight)
    # add edges from src to root nodes
    for node in root_nodes:
        if node == SRC: # in case there was already a src node
            continue
        task_graph.add_edge(SRC, node)
        task_graph[SRC][node]["weight"] = zero_weight
    # add edges from leaf nodes to dst
    for node in leaf_nodes:
        if node == DST: # in case there was already a dst node
            continue
        task_graph.add_edge(node, DST)
        task_graph[node][DST]["weight"] = zero_weight

    return network, task_graph
