import logging
from typing import Dict, Hashable, List, Tuple
import networkx as nx
import numpy as np

from saga.utils.tools import get_insert_loc

from ..base import Scheduler, Task
from ..utils.random_variable import RandomVariable


def stoch_heft_rank_sort(network: nx.Graph, 
                         task_graph: nx.DiGraph,
                         runtimes: Dict[Hashable, Dict[Hashable, float]],
                         commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]) -> List[Hashable]:
    """Sorts the tasks in the task graph based on the expected runtime.

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.
        runtimes (Dict[Hashable, Dict[Hashable, float]]): A dictionary mapping nodes to a dictionary of tasks and their runtimes.
        commtimes (Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]): A dictionary mapping edges to a dictionary of task dependencies and their communication times.

    Returns:
        List[Hashable]: A list of tasks sorted by their expected runtime.
    """
    rank = {}
    logging.debug(f"Topological sort: {list(nx.topological_sort(task_graph))}")
    for task_name in reversed(list(nx.topological_sort(task_graph))):
        avg_comp = np.mean([runtimes[node][task_name] for node in network.nodes])
        max_comm = 0 if task_graph.out_degree(task_name) <= 0 else max(
            ( 
                rank.get(succ, 0) + 
                np.mean([commtimes[src, dst][task_name] for src, dst in network.edges])
            )
            for succ in task_graph.successors(task_name)
        )
        rank[task_name] = avg_comp + max_comm

    return sorted(list(rank.keys()), key=rank.get, reverse=True)

class StochHeftScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_runtimes(network: nx.Graph, task_graph: nx.DiGraph) -> Tuple[Dict[Hashable, Dict[Hashable, float]], Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]]:
        """Get the runtimes of all tasks on all nodes.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Tuple[Dict[Hashable, Dict[Hashable, float]], Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]]: A tuple of dictionaries mapping nodes to a dictionary of tasks and their runtimes and edges to a dictionary of tasks and their communication times.
                The first dictionary maps nodes to a dictionary of tasks and their runtimes.
                The second dictionary maps edges to a dictionary of task dependencies and their communication times.
        """
        runtimes = {}
        for node in network.nodes:
            runtimes[node] = {}
            speed: RandomVariable = network.nodes[node]["weight"]
            exp_speed = speed.mean()
            for task in task_graph.nodes:
                cost: RandomVariable = task_graph.nodes[task]["weight"]
                exp_cost = cost.mean()
                runtimes[node][task] = exp_cost / exp_speed

        commtimes = {}
        for src, dst in network.edges:
            commtimes[src, dst] = {}
            speed: RandomVariable = network.edges[src, dst]["weight"]
            exp_speed = speed.mean()
            for src_task, dst_task in task_graph.edges:
                cost = task_graph.edges[src_task, dst_task]["weight"]
                exp_cost = cost.mean()
                commtimes[src, dst][src_task, dst_task] = exp_cost / exp_speed

        return runtimes, commtimes

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Schedules all tasks on the node with the highest processing speed

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: A schedule mapping nodes to a list of tasks.

        Raises:
            ValueError: If the instance is not valid
        """
        runtimes, commtimes = self.get_runtimes(network, task_graph)
        
        comp_schedule: Dict[str, List[Task]] = {node: [] for node in network.nodes}
        task_schedule: Dict[str, Task] = {}

        task_eft_rvs: Dict[str, RandomVariable] = {}
        task_name: str
        for task_name in stoch_heft_rank_sort(network, task_graph, runtimes, commtimes):
            task_size = task_graph.nodes[task_name]["weight"] 

            candidate_tasks: List[Tuple[int, Task]] = []
            for node in network.nodes: # Find the best node to run the task
                node_speed = network.nodes[node]["weight"] 
                exp_max_arrival_time = 0
                if task_graph.in_degree(task_name) > 0:
                    max_arrival_time = RandomVariable.max(
                        *[
                            # TODO: this isn't right we should use join dist X/Y not E[X]/E[Y]
                            task_eft_rvs[parent] + # commtimes[task_schedule[parent].node, node][parent, task_name]
                            (task_graph.edges[parent, task_name]["weight"] / network.edges[task_schedule[parent].node, node]["weight"])
                            for parent in task_graph.predecessors(task_name)
                        ]
                    )
                    exp_max_arrival_time = max_arrival_time.mean()

                _idx, exp_start_time = get_insert_loc(comp_schedule[node], exp_max_arrival_time, task_size / node_speed)
                exp_end_time = exp_start_time + runtimes[node][task_name]
                candidate_task = Task(node, task_name, exp_start_time, exp_end_time)
                candidate_tasks.append((_idx, candidate_task))

            idx, new_task = min(candidate_tasks, key=lambda x: x[1].exp_end_time)

            comp_schedule[new_task.node].insert(idx, new_task)
            task_schedule[new_task.name] = new_task

        return comp_schedule
        