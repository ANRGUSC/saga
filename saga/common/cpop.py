import logging
from typing import Dict, Hashable, List, Tuple
import networkx as nx
import numpy as np
import heapq

from ..utils.tools import get_insert_loc, check_instance_simple
from ..base import Scheduler, Task

def cpop_ranks(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """Computes the ranks of the tasks in the task graph using for the CPoP algorithm.

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.
    
    Returns:
        Dict[Hashable, float]: The ranks of the tasks in the task graph.
            Keys are task names and values are the ranks.
    """
    rank_up = {}
    top_sort = list(nx.topological_sort(task_graph))
    for task_name in reversed(top_sort):
        avg_comp = np.mean([
            task_graph.nodes[task_name]['weight'] / 
            network.nodes[node]['weight'] for node in network.nodes
        ])
        max_comm = 0 if task_graph.out_degree(task_name) <= 0 else max(
            ( 
                rank_up[succ] + # rank of successor
                np.mean([ # average communication time for input data of successor
                    task_graph.edges[task_name, succ]['weight'] /
                    network.edges[src, dst]['weight'] for src, dst in network.edges
                ])
            )
            for succ in task_graph.successors(task_name)
        )
        rank_up[task_name] = avg_comp + max_comm 

    rank_down = {}
    # rank_down[i] = max_{j in pred(i)} (rank_down[j] + avg_comp_j + avg_comm_{j,i})
    for task_name in top_sort:
        max_comm = 0 if task_graph.in_degree(task_name) <= 0 else max(
            ( 
                rank_down[pred] + # rank of predecessor
                np.mean([ # average computation time of predecessor
                    task_graph.nodes[pred]['weight'] / 
                    network.nodes[node]['weight'] for node in network.nodes
                ]) +
                np.mean([ # average communication time for output data of predecessor
                    task_graph.edges[pred, task_name]['weight'] /
                    network.edges[src, dst]['weight'] for src, dst in network.edges
                ])
            )
            for pred in task_graph.predecessors(task_name)
        )
        rank_down[task_name] = avg_comp + max_comm

    rank = {task_name: rank_up[task_name] + rank_down[task_name] for task_name in task_graph.nodes}
    return rank

class CPOPScheduler(Scheduler):
    """Implements the CPoP algorithm for task scheduling.

    Attributes:
        name (str): The name of the scheduler.
    """
    def __init__(self) -> None:
        super(CPOPScheduler, self).__init__()

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
        logging.debug(f"ranks: {ranks}")

        start_task = next(task for task in task_graph.nodes if task_graph.in_degree(task) == 0)
        end_task = next(task for task in task_graph.nodes if task_graph.out_degree(task) == 0)
        # cp_rank is rank of tasks on critical path (rank of start task)
        cp_rank = ranks[start_task]
        logging.debug(f"CP rank: {cp_rank}")

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

        pq = [(ranks[start_task], start_task)]
        heapq.heapify(pq)
        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_map: Dict[Hashable, Task] = {}
        while pq:
            # get highest priority task
            task_rank, task_name = heapq.heappop(pq)
            logging.debug(f"Processing task {task_name} (predecessors: {list(task_graph.predecessors(task_name))})")
            
            if np.isclose(task_rank, cp_rank):
                # assign task to cp_node
                node = cp_node
                start_time = 0 if not schedule[node] else schedule[node][-1].end
                end_time = start_time + task_graph.nodes[task_name]["weight"] / network.nodes[node]["weight"]
                schedule[node].append(Task(node, task_name, start_time, end_time))
                task_map[task_name] = schedule[node][-1]
            else:
                # schedule on node with earliest completion time
                candidate_tasks: Dict[Hashable, Tuple[int, Task]] = {}
                for node in network.nodes:
                    parent_data_arrival_times = {
                        parent: task_map[parent].end + (
                            0 if task_map[parent].node == node else
                            task_graph.edges[parent, task_name]["weight"] / network.edges[task_map[parent].node, node]["weight"]
                        )
                        for parent in task_graph.predecessors(task_name)
                    }
                    min_start_time = max(parent_data_arrival_times.values()) if parent_data_arrival_times else 0
                    exec_time = task_graph.nodes[task_name]["weight"] / network.nodes[node]["weight"]
                    idx, start_time = get_insert_loc(schedule[node], min_start_time, exec_time)
                    end_time = start_time + exec_time
                    candidate_tasks[node] = idx, Task(node, task_name, start_time, end_time)
                idx, new_task = min(candidate_tasks.values(), key=lambda x: x[1].end)
                schedule[new_task.node].insert(idx, new_task)
                task_map[task_name] = new_task

            # get ready tasks
            ready_tasks = [
                succ for succ in task_graph.successors(task_name)
                if all(pred in task_map for pred in task_graph.predecessors(succ))
            ]
            for ready_task in ready_tasks:
                heapq.heappush(pq, (ranks[ready_task], ready_task))

        return schedule


            





        
