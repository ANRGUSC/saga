from typing import Dict, Hashable, List, Tuple
import networkx as nx
import numpy as np
from .utils import get_ready_time, get_insert_loc, insert
from saga.scheduler import Task
from . import get_earliest_finish_time_insert
def critical_path_insert_schedule(
        network: nx.Graph,
        task_graph: nx.DiGraph,
        runtimes: Dict[Hashable, Dict[Hashable, float]],
        commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
        comp_schedule: Dict[Hashable, List[Task]],
        task_schedule: Dict[Hashable, Task],
        task_name: Hashable,
        priority: int,
        ) -> None:
    

    if priority == 1:

        critical_node = max(network.nodes, key=lambda node: network.nodes[node]['weight'])
        insert(task_graph, runtimes, commtimes, critical_node, task_name, comp_schedule, task_schedule)
    
    else:
        min_finish_time = np.inf
        for node in network.nodes:  # Find the best node to run the task
            finish_time = get_earliest_finish_time_insert(task_graph, runtimes, commtimes, node, task_name, comp_schedule, task_schedule)
            if finish_time < min_finish_time:
                min_finish_time = finish_time
                best_node = node
        insert(task_graph, runtimes, commtimes, best_node, task_name, comp_schedule, task_schedule)