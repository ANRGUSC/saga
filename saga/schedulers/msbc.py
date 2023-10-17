import heapq
import logging
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np

from ..scheduler import Scheduler, Task
from ..utils.tools import get_insert_loc


def calulate_sbct(network: nx.Graph,
                  task_graph: nx.DiGraph,
                  runtimes: Dict[Hashable, Dict[Hashable, float]],
                  commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]) -> (Dict[Hashable, float], Dict[Hashable, Hashable]):
    sbct = {}
    ifav = {}

    def get_drt(task_name, node):

        return max(
            (sbct[pred] + commtimes[ifav[pred], node][pred, task_name]) 
            for pred in task_graph.predecessors(task_name)
        )

    def get_sbct(task_name):
        min_val = float("inf")
        min_node = None
        degree = task_graph.in_degree(task_name)
        for node in network.nodes:
            # print("degree",degree)
            if degree <= 0:
                temp_val = runtimes[node][task_name]
            else:
                temp_val = get_drt(task_name, node) + runtimes[node][task_name]

            if temp_val<min_val:
                min_val = temp_val
                min_node = node
        return min_val, min_node

    for task_name in nx.topological_sort(task_graph):
        sbct[task_name], ifav[task_name] = get_sbct(task_name)
        # print(task_name, sbct[task_name], ifav[task_name])

    return sbct, ifav

def get_sbl(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """Computes the downward rank of the tasks in the task graph."""
    sbl = {}

    is_comp_zero = all(np.isclose(network.nodes[_node]['weight'], 0) for _node in network.nodes)

    def avg_comp_time(task: Hashable) -> float:
        if is_comp_zero:
            return 1e-9
        return np.mean([
            task_graph.nodes[task]['weight'] / network.nodes[node]['weight']
            for node in network.nodes
            if not np.isclose(network.nodes[node]['weight'], 0)
        ])

    for task_name in nx.topological_sort(task_graph):
        sbl[task_name] = 0 if task_graph.in_degree(task_name) <= 0 else max(
            (
                avg_comp_time(pred) + sbl[pred]
            )
            for pred in task_graph.predecessors(task_name)
        )

    return sbl

def calculate_st(task_graph: nx.DiGraph, ifav: Dict[Hashable, float],
                 runtimes: Dict[Hashable, Dict[Hashable, float]],
                 commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]) -> Dict[Hashable, float]:
    st = {}
    for task_name in nx.topological_sort(task_graph):
        st[task_name] = 0 if task_graph.in_degree(task_name) <= 0 else max(
            (
                st[pred] + runtimes[ifav[pred]][pred] + commtimes[ifav[pred],ifav[task_name]][pred,task_name]
            )
            for pred in task_graph.predecessors(task_name)
        )
    return st

def get_priority(task_graph: nx.Graph, sbct: Dict[Hashable,float], sbl: Dict[Hashable, float], st: Dict[Hashable, float]) -> Dict[Hashable, float]:
    return  {task_name: sbct[task_name] + sbl[task_name] + st[task_name] for task_name in task_graph.nodes}

class MsbcScheduler(Scheduler): # pylint: disable=too-few-public-methods
    """Implements the CPoP algorithm for task scheduling.
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





    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Computes the schedule for the task graph using the MSBC algorithm.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[str, List[Task]]: The schedule for the task graph.

        Raises:
            ValueError: If instance is invalid.
        """
        runtimes, commtimes = MsbcScheduler.get_runtimes(network, task_graph)

        sbl = get_sbl(network,task_graph)

        sbct, ifav = calulate_sbct(network, task_graph, runtimes, commtimes)

        st = calculate_st(task_graph, ifav, runtimes, commtimes)

        priorities = get_priority(task_graph, sbct, sbl, st)
        print("SBL:", sbl)
        print("SBCT:", sbct)
        print("ifav", ifav)
        print("st", st)
        print(priorities)
        return None