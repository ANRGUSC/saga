from typing import Dict, Hashable, List, Tuple, Optional

import networkx as nx
import numpy as np
import logging

from ..utils.tools import get_insert_loc
from ..scheduler import Scheduler, Task
from .heft import HeftScheduler
from .cpop import upward_rank

class LAHeftScheduler(HeftScheduler):

    '''The options for mode could be 'weighted_average' or 'maximum'.'''
    def __init__(self, mode: Optional[str] = 'weighted_average', priority_change: Optional[bool] = False) -> None:
        super().__init__()
        self.mode = mode
        self.priority_change = priority_change

    def _schedule(self,
                  network: nx.Graph,
                  task_graph: nx.DiGraph,
                  runtimes: Dict[Hashable, Dict[Hashable, float]],
                  commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
                  schedule_order: List[Hashable]) -> Dict[Hashable, List[Task]]:
        if self.priority_change:
            return self._schedule_with_priority_change(network, task_graph, runtimes, commtimes, schedule_order)
        else:
            return self._schedule_without_priority_change(network, task_graph, runtimes, commtimes, schedule_order)

    def _schedule_with_priority_change(self,
                  network: nx.Graph,
                  task_graph: nx.DiGraph,
                  runtimes: Dict[Hashable, Dict[Hashable, float]],
                  commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
                  schedule_order: List[Hashable]) -> Dict[Hashable, List[Task]]:
        comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}
        rank = upward_rank(network, task_graph)
        task_name: Hashable
        logging.debug("Schedule order: %s", schedule_order)
        schedule_order_copy = schedule_order.copy()
        while schedule_order_copy:
            task1 = schedule_order_copy.pop(0)
            if schedule_order_copy:
                task2 = schedule_order_copy.pop(0)
                min_finish_time_overall = np.inf
                min_finish_task = None
                min_finish_task_best_node = None
                for task_name in [task1,task2]:
                    children= set(task_graph.successors(task1)).union(set(task_graph.successors(task2))).union({task1} if task2==task_name else {task2})
                    logging.debug("Children is %s", str(children))
                    best_node, min_finish_time_task = self.find_best_node(network,task_schedule,task_graph,runtimes,commtimes,task_name,comp_schedule,schedule_order,rank,children)
                    if min_finish_time_task < min_finish_time_overall:
                        min_finish_time_overall = min_finish_time_task
                        min_finish_task = task_name
                        min_finish_task_best_node = best_node
                task_name = min_finish_task
                best_node = min_finish_task_best_node
                min_finish_time = min_finish_time_overall
                new_runtime = runtimes[best_node[0]][task_name]
                task = Task(best_node[0], task_name, min_finish_time - new_runtime, min_finish_time)
                comp_schedule[best_node[0]].insert(best_node[1], task)
                task_schedule[task_name] = task
                logging.debug("Scheduled task %s on node %s: start time %s, finish time %s", task_name, best_node[0], task.start, task.end)
                schedule_order_copy.insert(0,task1 if task2==task_name else task2)
                continue
            else:
                task_name = task1
                best_node, min_finish_time = self.find_best_node(network,task_schedule,task_graph,runtimes,commtimes,task_name,comp_schedule,schedule_order,rank,task_graph.successors(task_name))
                new_runtime = runtimes[best_node[0]][task_name]
                task = Task(best_node[0], task_name, min_finish_time - new_runtime, min_finish_time)
                comp_schedule[best_node[0]].insert(best_node[1], task)
                task_schedule[task_name] = task
                logging.debug("Scheduled task %s on node %s: start time %s, finish time %s", task_name, best_node[0], task.start, task.end)

        return comp_schedule
            
        
                    

    def find_best_node(self,network,task_schedule,task_graph,runtimes,commtimes,task_name,comp_schedule,schedule_order,rank,children):
        min_finish_time = np.inf
        min_child_finish_time = np.inf
        best_node = None
        for node in network.nodes: # Find the best node to run the task
            max_arrival_time: float = max( #
                [
                    0.0, *[
                        task_schedule[parent].end + (
                            commtimes[(task_schedule[parent].node, node)][(parent, task_name)]
                        )
                        for parent in task_graph.predecessors(task_name) if parent in task_schedule
                    ]
                ]
            )

            runtime = runtimes[node][task_name]
            idx, start_time = get_insert_loc(comp_schedule[node], max_arrival_time, runtime)
            
            logging.debug("Testing task %s on node %s: start time %s, finish time %s", task_name, node, start_time, start_time + runtime)

            finish_time = start_time + runtime
            
            child_finish_time_factor = 0.0
            if(self.mode == 'weighted_average'):
                rank_sum = 0.0
            for child in schedule_order:
                if child in children:
                    child_min_finish_time = np.inf
                    child_best_node = None
                    for child_node in network.nodes:
                        child_max_arrival_time: float = max(
                            [
                                0.0, finish_time + (commtimes[node, child_node][task_name, child]) if (child in task_graph.successors(task_name)) else 0.0, *[
                                    task_schedule[parent].end + (
                                        commtimes[(task_schedule[parent].node, child_node)][(parent, child)]
                                    )
                                    for parent in task_graph.predecessors(child) if parent in task_schedule
                                ]
                                
                            ]
                        )
                        child_runtime = runtimes[child_node][child]
                        child_idx, child_start_time = get_insert_loc(comp_schedule[child_node], child_max_arrival_time, child_runtime)

                        child_finish_time = child_start_time + child_runtime
                        if child_finish_time < child_min_finish_time:
                            child_min_finish_time = child_finish_time
                            child_best_node = child_node, child_idx
                    match self.mode:
                        case 'weighted_average':
                            child_finish_time_factor += child_min_finish_time * rank[child]
                            rank_sum += rank[child]
                        case 'maximum':
                            child_finish_time_factor = max(child_finish_time_factor, child_min_finish_time)
            if child_finish_time_factor != 0.0:
                if(self.mode == 'weighted_average'):
                    child_finish_time_factor /= rank_sum
            else:
                child_finish_time_factor = finish_time
            
            logging.debug("After the child factor testing Task %s on node %s: finish time %s", task_name, node, finish_time)
            if child_finish_time_factor < min_child_finish_time:
                min_child_finish_time = child_finish_time_factor
                min_finish_time = finish_time
                best_node = node, idx
        return best_node, min_finish_time

    def _schedule_without_priority_change(self,
                  network: nx.Graph,
                  task_graph: nx.DiGraph,
                  runtimes: Dict[Hashable, Dict[Hashable, float]],
                  commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
                  schedule_order: List[Hashable]) -> Dict[Hashable, List[Task]]:
        comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}
        rank = upward_rank(network, task_graph)
        task_name: Hashable
        logging.debug("Schedule order: %s", schedule_order)
        for task_name in schedule_order:
            best_node, min_finish_time = self.find_best_node(network,task_schedule,task_graph,runtimes,commtimes,task_name,comp_schedule,schedule_order,rank,task_graph.successors(task_name))
            new_runtime = runtimes[best_node[0]][task_name]
            task = Task(best_node[0], task_name, min_finish_time - new_runtime, min_finish_time)
            comp_schedule[best_node[0]].insert(best_node[1], task)
            task_schedule[task_name] = task

        return comp_schedule
        
        