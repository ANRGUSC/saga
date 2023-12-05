import logging
import pprint
from typing import Dict, Hashable, List, Tuple, Union

import networkx as nx
import numpy as np

from saga.utils.random_variable import RandomVariable
from saga.utils.tools import get_insert_loc

from ...scheduler import Scheduler, Task


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
                np.mean([commtimes[src, dst][task_name, succ] for src, dst in network.edges])
            )
            for succ in task_graph.successors(task_name)
        )
        rank[task_name] = avg_comp + max_comm

    sorted_tasks = sorted(list(rank.keys()), key=rank.get, reverse=True)
    logging.debug(f"Task Ranks: {rank}")
    return sorted_tasks

class StochHeftScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_runtimes(network: nx.Graph, task_graph: nx.DiGraph) -> Tuple[Dict[Hashable, Dict[Hashable, float]], Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]]:
        """Get the expected runtimes of all tasks on all nodes.

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
            logging.debug(f"Node {node} has speed {speed}")
            assert(isinstance(speed, RandomVariable))
            for task in task_graph.nodes:
                cost: RandomVariable = task_graph.nodes[task]["weight"]
                logging.debug(f"Task {task} has cost {cost}")
                assert(isinstance(cost, RandomVariable))
                runtimes[node][task] = (cost / speed).mean()
                logging.debug(f"Task {task} on node {node} has expected runtime {runtimes[node][task]}")

        commtimes = {}
        for src, dst in network.edges:
            commtimes[src, dst] = {}
            commtimes[dst, src] = {}
            speed: RandomVariable = network.edges[src, dst]["weight"]
            logging.debug(f"Edge {src} -> {dst} has speed {speed}")
            assert(isinstance(speed, RandomVariable))
            for src_task, dst_task in task_graph.edges:
                cost = task_graph.edges[src_task, dst_task]["weight"]
                logging.debug(f"Edge {src_task} -> {dst_task} has cost {cost}")
                assert(isinstance(cost, RandomVariable))
                commtimes[src, dst][src_task, dst_task] = (cost / speed).mean()
                commtimes[dst, src][src_task, dst_task] = commtimes[src, dst][src_task, dst_task]

                logging.debug(f"Task {src_task} on node {src} to task {dst_task} on node {dst} has expected communication time {commtimes[src, dst][src_task, dst_task]}")

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
            task_size: RandomVariable = task_graph.nodes[task_name]["weight"] 

            candidate_tasks: List[Tuple[int, Task]] = []
            for node in network.nodes: # Find the best node to run the task
                node_speed: RandomVariable = network.nodes[node]["weight"] 
                exp_max_arrival_time = 0
                max_arrival_time: Union[float, RandomVariable] = 0.0
                if task_graph.in_degree(task_name) > 0:
                    arrival_times = [
                        task_eft_rvs[parent] + # EFT of parent
                        (task_graph.edges[parent, task_name]["weight"] / 
                            network.edges[task_schedule[parent].node, node]["weight"])
                        for parent in task_graph.predecessors(task_name)
                    ]
                    logging.debug(f"Arrival times: {pprint.pformat([t.mean() for t in arrival_times])} for task {task_name} on node {node}")
                    max_arrival_time = RandomVariable.max(*arrival_times)
                    exp_max_arrival_time = max_arrival_time.mean()
                    parent_finish_times = [(task_eft_rvs[parent].mean(), task_schedule[parent].end) for parent in task_graph.predecessors(task_name)]
                    parent_comm_times = [
                        (
                            (f"{parent} -> {task_name} from {task_schedule[parent].node} to {node}"),
                            (task_graph.edges[parent, task_name]["weight"] / network.edges[task_schedule[parent].node, node]["weight"]).mean() 
                        )
                        for parent in task_graph.predecessors(task_name)
                    ]
                    logging.debug(f"Parent finish times: {pprint.pformat(parent_finish_times)} for task {task_name} on node {node}")
                    logging.debug(f"Parent communication times: {pprint.pformat(parent_comm_times)} for task {task_name} on node {node}")
                    logging.debug(f"Task {task_name} on node {node} has expected max arrival time of parent data {exp_max_arrival_time}")

                _idx, start_time = get_insert_loc(comp_schedule[node], exp_max_arrival_time, runtimes[node][task_name])
                exp_end_time = start_time + runtimes[node][task_name]
                candidate_task = Task(node, task_name, start_time, exp_end_time)

                exp_delay = start_time - exp_max_arrival_time
                logging.debug(f"Task {task_name} on node {node} has expected delay {exp_delay}")
                # EFT of the task
                eft_rv = (task_size / node_speed) + (max_arrival_time + exp_delay)

                candidate_tasks.append((_idx, candidate_task, eft_rv))

            idx, new_task, eft_rv = min(candidate_tasks, key=lambda x: x[1].end)

            logging.debug(f"Check {eft_rv.mean()} ~= {new_task.end} for task {new_task.name} on node {new_task.node}")

            comp_schedule[new_task.node].insert(idx, new_task)
            task_schedule[new_task.name] = new_task
            task_eft_rvs[new_task.name] = eft_rv

        return comp_schedule
        