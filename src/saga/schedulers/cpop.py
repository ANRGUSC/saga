import heapq
import logging
from typing import Dict, Hashable, List

import networkx as nx
import numpy as np

from ..scheduler import Scheduler, Task
from ..utils.tools import get_insert_loc

def upward_rank(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """Computes the upward rank of the tasks in the task graph."""
    ranks = {}

    topological_order = list(nx.topological_sort(task_graph))
    for node in topological_order[::-1]:
        # rank = avg_comp_time + max(rank of successors + avg_comm_time w/ successors)
        avg_comp_time = np.mean([
            task_graph.nodes[node]['weight'] / network.nodes[neighbor]['weight']
            for neighbor in network.nodes
        ])
        max_comm_time = 0 if task_graph.out_degree(node) <= 0 else max(
            [
                ranks[neighbor] + np.mean([
                    task_graph.edges[node, neighbor]['weight'] / network.edges[src, dst]['weight']
                    for src, dst in network.edges
                ])
                for neighbor in task_graph.successors(node)
            ]
        )
        ranks[node] = avg_comp_time + max_comm_time

    return ranks

def downward_rank(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    ranks = {}

    topological_order = list(nx.topological_sort(task_graph))

    for node in topological_order:
        # rank = max(rank of predecessors + avg_comm_time w/ predecessors + avg_comp_time)
        ranks[node] = 0 if task_graph.in_degree(node) <= 0 else max(
            [
                ranks[pred] + np.mean([
                    task_graph.edges[pred, node]['weight'] / network.edges[src, dst]['weight']
                    for src, dst in network.edges
                ]) + np.mean([
                    task_graph.nodes[pred]['weight'] / network.nodes[neighbor]['weight']
                    for neighbor in network.nodes
                ])
                for pred in task_graph.predecessors(node)
            ]
        )

    return ranks

def cpop_ranks(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """Computes the ranks of the tasks in the task graph using for the CPoP algorithm.

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        Dict[Hashable, float]: The ranks of the tasks in the task graph.
            Keys are task names and values are the ranks.
    """
    upward_ranks = upward_rank(network, task_graph)
    downward_ranks = downward_rank(network, task_graph)
    return {
        task_name: (upward_ranks[task_name] + downward_ranks[task_name])
        for task_name in task_graph.nodes
    }

class CpopScheduler(Scheduler): # pylint: disable=too-few-public-methods
    """Implements the CPoP algorithm for task scheduling.

    Source: https://dx.doi.org/10.1109/71.993206
    """
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Computes the schedule for the task graph using the CPoP algorithm.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[str, List[Task]]: The schedule for the task graph.

        Raises:
            ValueError: If instance is invalid.
        """
        ranks = cpop_ranks(network, task_graph)
        logging.debug("Ranks: %s", ranks)

        start_task = next(task for task in task_graph.nodes if task_graph.in_degree(task) == 0)
        _ = next(task for task in task_graph.nodes if task_graph.out_degree(task) == 0)
        # cp_rank is rank of tasks on critical path (rank of start task)
        cp_rank = ranks[start_task]
        logging.debug("Critical path rank: %s", cp_rank)

        # node that minimizes sum of execution times of tasks on critical path
        # this should just be the node with the highest weight
        cp_node = min(
            network.nodes,
            key=lambda node: sum(
                task_graph.nodes[task]['weight'] / network.nodes[node]['weight']
                for task in task_graph.nodes
                if np.isclose(ranks[task], cp_rank)
            )
        )
        logging.debug("Critical path node: %s", cp_node)

        pq = [(-ranks[start_task], start_task)]
        heapq.heapify(pq)
        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_map: Dict[Hashable, Task] = {}
        while pq:
            # get highest priority task
            task_rank, task_name = heapq.heappop(pq)
            logging.debug("Processing task %s (predecessors: %s)", task_name, list(task_graph.predecessors(task_name)))

            min_finish_time = np.inf
            best_node, best_idx = None, None
            if np.isclose(-task_rank, cp_rank):
                # assign task to cp_node
                exec_time = task_graph.nodes[task_name]["weight"] / network.nodes[cp_node]["weight"]
                max_arrival_time: float = max( #
                    [
                        0.0, *[
                            task_map[parent].end + (
                                (task_graph.edges[parent, task_name]["weight"] /
                                 network.edges[task_map[parent].node, cp_node]["weight"])
                            )
                            for parent in task_graph.predecessors(task_name)
                        ]
                    ]
                )
                best_node = cp_node
                best_idx, start_time = get_insert_loc(schedule[cp_node], max_arrival_time, exec_time)
                min_finish_time = start_time + exec_time
            else:
                # schedule on node with earliest completion time
                for node in network.nodes:
                    max_arrival_time: float = max( #
                        [
                            0.0, *[
                                task_map[parent].end + (
                                    task_graph.edges[parent, task_name]["weight"] /
                                    network.edges[task_map[parent].node, node]["weight"]
                                )
                                for parent in task_graph.predecessors(task_name)
                            ]
                        ]
                    )
                    exec_time = task_graph.nodes[task_name]["weight"] / network.nodes[node]["weight"]
                    idx, start_time = get_insert_loc(schedule[node], max_arrival_time, exec_time)
                    end_time = start_time + exec_time
                    if end_time < min_finish_time:
                        min_finish_time = end_time
                        best_node, best_idx = node, idx

            new_exec_time = task_graph.nodes[task_name]["weight"] / network.nodes[best_node]["weight"]
            new_task = Task(best_node, task_name, min_finish_time - new_exec_time, min_finish_time)
            schedule[new_task.node].insert(best_idx, new_task)
            task_map[task_name] = new_task

            # get ready tasks
            ready_tasks = [
                succ for succ in task_graph.successors(task_name)
                if all(pred in task_map for pred in task_graph.predecessors(succ))
            ]
            for ready_task in ready_tasks:
                heapq.heappush(pq, (-ranks[ready_task], ready_task))

        return schedule
