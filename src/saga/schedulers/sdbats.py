import logging
import pathlib
from typing import Dict, Hashable, List

import networkx as nx
import numpy as np

from ..scheduler import Task
from .heft import HeftScheduler, heft_rank_sort

def upward_rank_std(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """Computes the upward rank of the tasks in the task graph based on standard deviation."""
    ranks = {}

    topological_order = list(nx.topological_sort(task_graph))
    for node in topological_order[::-1]:
        # rank = std_comp_time + max(rank of successors + std_comm_time w/ successors)
        avg_comp_time = np.std([
            task_graph.nodes[node]['weight'] / network.nodes[neighbor]['weight']
            for neighbor in network.nodes
        ])
        max_comm_time = 0 if task_graph.out_degree(node) <= 0 else max(
            [
                ranks[neighbor] + np.std([
                    task_graph.edges[node, neighbor]['weight'] / network.edges[src, dst]['weight']
                    for src, dst in network.edges
                ])
                for neighbor in task_graph.successors(node)
            ]
        )
        ranks[node] = avg_comp_time + max_comm_time

    return ranks

class SDBATSScheduler(HeftScheduler):

    """Schedules tasks using the SDBATS algorithm. Inherited from HeftScheduler
    """

    def schedule(
        self, network: nx.Graph, task_graph: nx.DiGraph
    ) -> Dict[str, List[Task]]:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[str, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """
        # get runtimes and communication times
        runtimes, commtimes = SDBATSScheduler.get_runtimes(network, task_graph)

        # get order of scheduling based on upward ranking
        schedule_order = heft_rank_sort(network, task_graph)

        # return the scheduled result
        return self._schedule(network, task_graph, runtimes, commtimes, schedule_order)
