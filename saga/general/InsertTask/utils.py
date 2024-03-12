from typing import Dict, Hashable, List, Tuple
import networkx as nx
from saga.scheduler import Task


def insert(task_graph:nx.DiGraph, 
           runtimes: Dict[Hashable, Dict[Hashable, float]], 
           commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]], 
           node: Hashable, 
           task_name: Hashable, 
           comp_schedule: Dict[Hashable, List[Task]], 
           task_schedule: Dict[Hashable, Task] ) -> None:
        runtime = runtimes[node][task_name]
        max_arrival_time = get_ready_time(node, task_name, task_graph, commtimes, task_schedule)
        
        idx, start_time = get_insert_loc(
            comp_schedule[node], max_arrival_time, runtime
        )
        task = Task(node, task_name, start_time, start_time + runtime)
        comp_schedule[node].insert(idx, task)
        task_schedule[task_name] = task

def get_ready_time(node, task_name, task_graph, commtimes, task_schedule):
    return  max(  #
                    [
                        0.0,
                        *[
                            task_schedule[parent].end
                            + (
                                commtimes[(task_schedule[parent].node, node)][
                                    (parent, task_name)
                                ]
                            )
                            for parent in task_graph.predecessors(task_name)
                        ],
                    ]
                )


def get_insert_loc(schedule: List[Task], 
                   min_start_time: float, 
                   exec_time: float) -> Tuple[int, float]:
    """Get the location where the task should be inserted in the list of tasks.
    
    Args:
        schedule (List[Task]): The list of scheduled tasks.
        min_start_time (float): The minimum start time of the task.
        exec_time (float): The execution time of the task.
        
    Returns:
        int: The index where the task should be inserted.
    """
    if not schedule or min_start_time + exec_time < schedule[0].start:
        return 0, min_start_time
    
    for i, (left, right) in enumerate(zip(schedule, schedule[1:]), start=1):
        if min_start_time >= left.end and min_start_time + exec_time <= right.start:
            return i, min_start_time
        elif min_start_time < left.end and left.end + exec_time <= right.start:
            return i, left.end

    return len(schedule), max(min_start_time, schedule[-1].end)