import logging
import pathlib
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np

from ..base import Task
from ..base import Scheduler
from ..utils.tools import check_instance_simple, get_insert_loc

thisdir = pathlib.Path(__file__).resolve().parent

def heft_rank_sort(network: nx.Graph, task_graph: nx.DiGraph) -> List[Hashable]:
    """Sort tasks based on their rank (as defined in the HEFT paper).
    
    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        List[Hashable]: The sorted list of tasks.
    """
    rank = {}
    logging.debug(f"Topological sort: {list(nx.topological_sort(task_graph))}")
    for task_name in reversed(list(nx.topological_sort(task_graph))):
        avg_comp = np.mean([
            task_graph.nodes[task_name]['weight'] / 
            network.nodes[node]['weight'] for node in network.nodes
        ])
        max_comm = 0 if task_graph.out_degree(task_name) <= 0 else max(
            ( 
                rank.get(succ, 0) + 
                np.mean([
                    task_graph.edges[task_name, succ]['weight'] /
                    network.edges[src, dst]['weight'] for src, dst in network.edges
                ])
            )
            for succ in task_graph.successors(task_name)
        )
        rank[task_name] = avg_comp + max_comm

    return sorted(list(rank.keys()), key=rank.get, reverse=True)


class HeftScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[str, List[Task]]:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[str, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """
        check_instance_simple(network, task_graph)

        comp_schedule: Dict[str, List[Task]] = {node: [] for node in network.nodes}
        task_schedule: Dict[str, Task] = {}

        task_name: str
        for task_name in heft_rank_sort(network, task_graph):
            task_size = task_graph.nodes[task_name]["weight"] 

            min_finish_time = np.inf 
            best_node = None 
            for node in network.nodes: # Find the best node to run the task
                node_speed = network.nodes[node]["weight"] 
                max_arrival_time: float = max( # 
                    [
                        0.0, *[
                            task_schedule[parent].end + (
                                (
                                    (
                                        task_graph.edges[(parent, task_name)]["weight"] / 
                                        network.edges[(task_schedule[parent].node, node)]["weight"]
                                    )  if node != task_schedule[parent].node else 0 # instantaneous communication if on the same node
                                )
                            )
                            for parent in task_graph.predecessors(task_name)
                        ]
                    ]
                )
                    
                idx, start_time = get_insert_loc(comp_schedule[node], max_arrival_time, task_size / node_speed)
                
                finish_time = start_time + (task_size / node_speed)
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_node = node, idx 
            
            task = Task(best_node[0], task_name, min_finish_time - task_size / network.nodes[best_node[0]]["weight"], min_finish_time)
            comp_schedule[best_node[0]].insert(best_node[1], task)
            task_schedule[task_name] = task

        return comp_schedule