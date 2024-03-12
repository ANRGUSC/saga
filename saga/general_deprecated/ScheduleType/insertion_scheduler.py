from typing import Dict, Hashable, List, Tuple
from saga.scheduler import Task
from .schedule_type import ScheduleType
from .utils import get_insert_loc, get_ready_time
import networkx as nx

class InsertionScheduler(ScheduleType):
    
    def __init__(self, task_graph:nx.DiGraph, 
                 runtimes:Dict[Hashable, Dict[Hashable, float]], 
                 commtimes: Dict[
                    Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]
                ],):
        self.task_graph = task_graph
        self.runtimes = runtimes
        self.commtimes = commtimes

    def insert(self, node: Hashable, task_name: Hashable, comp_schedule: Dict[Hashable, List[Task]], task_schedule: Dict[Hashable, Task] ) -> None:
        runtime = self.runtimes[node][task_name]
        max_arrival_time = get_ready_time(node, task_name, self.task_graph, self.commtimes, task_schedule)
        
        idx, start_time = get_insert_loc(
            comp_schedule[node], max_arrival_time, runtime
        )
        task = Task(node, task_name, start_time, start_time + runtime)
        comp_schedule[node].insert(idx, task)
        task_schedule[task_name] = task

    
    def get_earliest_finish(self, node: Hashable, task_name: Hashable, comp_schedule: Dict[Hashable, List[Task]], task_schedule: Dict[Hashable, Task]) -> float:
        runtime = self.runtimes[node][task_name]
        max_arrival_time = get_ready_time(node, task_name, self.task_graph, self.commtimes, task_schedule)
        
        _, start_time = get_insert_loc(
            comp_schedule[node], max_arrival_time, runtime
        )
        return start_time + runtime

    
    def get_earliest_start(self, node: Hashable, task_name: Hashable, comp_schedule: Dict[Hashable, List[Task]], task_schedule: Dict[Hashable, Task]) ->float:
        # print(self.commtimes.keys())
        runtime = self.runtimes[node][task_name]
        max_arrival_time = get_ready_time(node, task_name, self.task_graph, self.commtimes, task_schedule)
        _, start_time = get_insert_loc(
            comp_schedule[node], max_arrival_time, runtime
        )
        return start_time