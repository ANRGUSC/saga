from typing import Dict, Hashable, List
from saga.scheduler import Task
from schedule_type import ScheduleType
from .utils import get_insert_loc
import networkx as nx

class InsertionScheduler(ScheduleType):
    
    def __init__(self, taskgraph:nx.DiGraph, runtimes, commtimes):
        self.taskgraph = taskgraph
        self.runtimes = runtimes
        self.commtimes = commtimes

    def insert(self, node: Hashable, task_name: Hashable, comp_schedule: Dict[Hashable, List[Task]], task_schedule: Dict[Hashable, Task] ) -> None:
        runtime = self.runtimes[node][task_name]
        max_arrival_time = self.get_earliest_start(node, task_name, comp_schedule, task_schedule)
        
        idx, start_time = get_insert_loc(
            comp_schedule[node], max_arrival_time, runtime
        )

        comp_schedule[node].insert(idx, Task(task_name, node, start_time, start_time + runtime))
        task_schedule[task_name] = Task(task_name, node, start_time, start_time + runtime)

    
    def get_earliest_finish(self, node: Hashable, task_name: Hashable, comp_schedule: Dict[Hashable, List[Task]], task_schedule: Dict[Hashable, Task]) -> float:
        return self.get_earliest_start(node, task_name, comp_schedule, task_schedule) + self.runtimes[node][task_name]        


    def get_earliest_start(self, node: Hashable, task_name: Hashable, comp_schedule: Dict[Hashable, List[Task]], task_schedule: Dict[Hashable, Task]) ->float:
        
        return max(  #
            [
                0.0,
                *[
                    task_schedule[parent].end
                    + (
                        self.commtimes[(task_schedule[parent].node, node)][
                            (parent, task_name)
                        ]
                    )
                    for parent in self.task_graph.predecessors(task_name)
                ],
            ]
        )