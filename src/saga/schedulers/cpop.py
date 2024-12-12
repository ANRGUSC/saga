import heapq
import logging
from typing import Callable, Dict, Hashable, List

import networkx as nx
import numpy as np

from ..scheduler import Scheduler, Task
from ..utils.tools import get_insert_loc

def upward_rank(network: nx.Graph,
                task_graph: nx.DiGraph,
                transcript_callback: Callable[[str], None] = lambda x: x) -> Dict[Hashable, float]:
    """Computes the upward rank of the tasks in the task graph."""
    ranks = {}

    topological_order = list(nx.topological_sort(task_graph))
    for node in topological_order[::-1]:
        # rank = avg_comp_time + max(rank of successors + avg_comm_time w/ successors)
        avg_comp_time = np.mean([
            task_graph.nodes[node]['weight'] / network.nodes[neighbor]['weight']
            for neighbor in network.nodes
        ])
        transcript_callback(f"Task {node} avg comp time: {avg_comp_time:0.4f}")
        max_comm_time = 0 if task_graph.out_degree(node) <= 0 else max(
            [
                ranks[neighbor] + np.mean([
                    task_graph.edges[node, neighbor]['weight'] / network.edges[src, dst]['weight']
                    for src, dst in network.edges
                ])
                for neighbor in task_graph.successors(node)
            ]
        )
        transcript_callback(f"Task {node} max comm time: {max_comm_time:0.4f}")
        ranks[node] = avg_comp_time + max_comm_time
        transcript_callback(f"Task {node} rank: {ranks[node]:0.4f}")

    return ranks

def downward_rank(network: nx.Graph,
                  task_graph: nx.DiGraph,
                  transcript_callback: Callable[[str], None] = lambda x: x) -> Dict[Hashable, float]:
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
        transcript_callback(f"Task {node} downward rank: {ranks[node]:0.4f}")

    return ranks

def cpop_ranks(network: nx.Graph, task_graph: nx.DiGraph, transcript_callback: Callable[[str], None] = lambda x: x) -> Dict[Hashable, float]:
    """Computes the ranks of the tasks in the task graph using for the CPoP algorithm.

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        Dict[Hashable, float]: The ranks of the tasks in the task graph.
            Keys are task names and values are the ranks.
    """
    upward_ranks = upward_rank(network, task_graph)
    transcript_callback(f"Upward ranks: {upward_ranks}")
    downward_ranks = downward_rank(network, task_graph)
    transcript_callback(f"Downward ranks: {downward_ranks}")
    ranks = {
        task_name: (upward_ranks[task_name] + downward_ranks[task_name])
        for task_name in task_graph.nodes
    }
    transcript_callback(f"CPoP ranks: {ranks}")
    return ranks

class CpopScheduler(Scheduler): # pylint: disable=too-few-public-methods
    """Implements the CPoP algorithm for task scheduling.

    Source: https://dx.doi.org/10.1109/71.993206
    """
    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 transcript_callback: Callable[[str], None] = lambda x: x) -> Dict[str, List[Task]]:
        """Computes the schedule for the task graph using the CPoP algorithm.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[str, List[Task]]: The schedule for the task graph.

        Raises:
            ValueError: If instance is invalid.
        """
        ranks = cpop_ranks(network, task_graph, transcript_callback)

        start_task = next(task for task in task_graph.nodes if task_graph.in_degree(task) == 0)
        _ = next(task for task in task_graph.nodes if task_graph.out_degree(task) == 0)
        # cp_rank is rank of tasks on critical path (rank of start task)
        cp_rank = ranks[start_task]
        transcript_callback(f"Critical path rank: {cp_rank}")

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
        transcript_callback(f"Critical path node (that which minimizes sum of execution times): {cp_node}")

        pq = [(-ranks[start_task], start_task)]
        heapq.heapify(pq)
        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_map: Dict[Hashable, Task] = {}
        while pq:
            # get highest priority task
            task_rank, task_name = heapq.heappop(pq)
            transcript_callback(f"Scheduling task {task_name} with rank {task_rank}")

            min_finish_time = np.inf
            best_node, best_idx = None, None
            if np.isclose(-task_rank, cp_rank):
                transcript_callback(f"Task {task_name} is on critical path")
                # assign task to cp_node
                exec_time = task_graph.nodes[task_name]["weight"] / network.nodes[cp_node]["weight"]
                transcript_callback(f"Task {task_name} execution time on node {cp_node}: {exec_time}")
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
                transcript_callback(f"All required predecessor data for task {task_name} would be available on node {cp_node} at {max_arrival_time:0.4f}")

                best_node = cp_node
                best_idx, start_time = get_insert_loc(schedule[cp_node], max_arrival_time, exec_time)
                transcript_callback(f"Earliest large enough slot on node {cp_node} is at {best_idx} at {start_time:0.4f}")
                min_finish_time = start_time + exec_time
                transcript_callback(f"Task {task_name} on node {cp_node} would finish at {min_finish_time:0.4f}")
            else:
                # schedule on node with earliest completion time
                transcript_callback(f"Task {task_name} is not on critical path")
                for node in network.nodes:
                    transcript_callback(f"Testing task {task_name} on node {node}")
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
                    transcript_callback(f"All required predecessor data for task {task_name} would be available on node {node} at {max_arrival_time:0.4f}")
                    exec_time = task_graph.nodes[task_name]["weight"] / network.nodes[node]["weight"]
                    transcript_callback(f"Task {task_name} execution time on node {node}: {exec_time}")
                    idx, start_time = get_insert_loc(schedule[node], max_arrival_time, exec_time)
                    transcript_callback(f"Earliest large enough slot on node {node} is at {idx} at {start_time:0.4f}")
                    end_time = start_time + exec_time
                    transcript_callback(f"Task {task_name} on node {node} would finish at {end_time:0.4f}")
                    if end_time < min_finish_time:
                        if best_node is not None:
                            transcript_callback(f"This is better than the previous best finish time of {min_finish_time:0.4f} (on node {best_node})")
                        min_finish_time = end_time
                        best_node, best_idx = node, idx
                    else:
                        if best_node is not None:
                            transcript_callback(f"This is worse than the previous best finish time of {min_finish_time:0.4f} (on node {best_node})")

            new_exec_time = task_graph.nodes[task_name]["weight"] / network.nodes[best_node]["weight"]
            new_task = Task(best_node, task_name, min_finish_time - new_exec_time, min_finish_time)
            transcript_callback(f"Inserting {new_task}")
            schedule[new_task.node].insert(best_idx, new_task)
            task_map[task_name] = new_task

            # get ready tasks
            ready_tasks = [
                succ for succ in task_graph.successors(task_name)
                if all(pred in task_map for pred in task_graph.predecessors(succ))
            ]
            transcript_callback(f"The following tasks are now able to be scheduled: {ready_tasks}")
            for ready_task in ready_tasks:
                transcript_callback(f"Pushing task {ready_task} into the scheduling queue with rank {ranks[ready_task]}")
                heapq.heappush(pq, (-ranks[ready_task], ready_task))

            transcript_callback(f"Current schedule: {schedule}")

        return schedule
