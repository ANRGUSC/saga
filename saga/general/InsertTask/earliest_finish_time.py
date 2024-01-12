from typing import Dict, Hashable, List, Tuple
import networkx as nx
import numpy as np
from .utils import get_ready_time, get_insert_loc, insert
from .insert_task import InsertTask
from saga.scheduler import Task

def get_earliest_finish_time_insert(task_graph: nx.DiGraph,
                        runtimes: Dict[Hashable, Dict[Hashable, float]],
                        commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
                        node: Hashable, 
                        task_name: Hashable, 
                        comp_schedule: Dict[Hashable, List[Task]], 
                        task_schedule: Dict[Hashable, Task]) -> float:
        
        runtime = runtimes[node][task_name]
        max_arrival_time = get_ready_time(node, task_name, task_graph, commtimes, task_schedule)
        
        _, start_time = get_insert_loc(
            comp_schedule[node], max_arrival_time, runtime
        )
        return start_time + runtime

class EarliestFinishTimeInsert(InsertTask):
    def __init__(self):
        pass
    def __call__(
        self,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        runtimes: Dict[Hashable, Dict[Hashable, float]],
        commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
        comp_schedule: Dict[Hashable, List[Task]],
        task_schedule: Dict[Hashable, Task],
        task_name: Hashable,
        priority: int,
        ) -> None:
        """
        Insert the task into the schedule at the earliest possible finish time.

        Args:
            network (nx.Graph): The network to schedule onto.
            task_graph (nx.DiGraph): The task graph to schedule.
            runtimes (Dict[Hashable, Dict[Hashable, float]]): The runtimes of all tasks on all nodes.
            commtimes (Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]): The communication times of all tasks on all edges.
            node (Hashable): The node to insert the task onto.
            task_name (Hashable): The task to insert.
            comp_schedule (Dict[Hashable, List[Task]]): The current schedule.
            task_schedule (Dict[Hashable, Task]): The current task schedule.

        Returns:
            None
        """
        min_finish_time = np.inf
        best_node = None
        for node in network.nodes:  # Find the best node to run the task
            finish_time = get_earliest_finish_time_insert(task_graph, runtimes, commtimes, node, task_name, comp_schedule, task_schedule)
            if finish_time < min_finish_time:
                min_finish_time = finish_time
                best_node = node
        insert(task_graph, runtimes, commtimes, best_node, task_name, comp_schedule, task_schedule)