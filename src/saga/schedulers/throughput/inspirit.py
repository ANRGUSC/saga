import pathlib
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, PrivateAttr, Field

from saga import Schedule, Scheduler, ScheduledTask, TaskGraph, Network, TaskGraphNode, NetworkNode, TaskGraphEdge
from saga.schedulers.cpop import upward_rank
import heapq





def compute_inspiring_effeciency(task_graph: TaskGraph, network: Network, time_window: float) -> Dict[str, float]:
    """Compute the inspiring efficiency of tasks in a task graph.
    A task's inspiring efficiency is the number of tasks expected to finish in a given time window.

    Args:
        task_graph (TaskGraph): The task graph to compute the inspiring efficiency for.
        network (Network): The network to compute the inspiring efficiency for.
        time_window (float): The time window to compute the inspiring efficiency for.

    Returns: 
       effeciency_ranks: A dict of task_name to inspiring efficiency, where a task's inspiring efficiency is the number of tasks expected to finish in the given time window.
    """
    average_network_speed = np.mean([node.speed for node in network.nodes])
    
    #the paper suggests using the average execution time of tasks of a certain type, but
    #because we don't have task types, we will just manually calculate excecution costs of 
    #of children tasks. This will be more computationally expensive, but can be easily optimized by task types.
    def compute_task_inspiring_ability(task) -> float:
        """Compute the inspiring ability of a task in a task graph.
        A task's inspiring ability is the total amount of tasks dependent on it that can be expected to finish in a given time window.
        
        Args:
            task (TaskGraphNode): The task to compute the inspiring ability for.
        Returns:
            float: The inspiring ability of the task.
        """
        count = 0
        # counter = 0
        time = 0
        current_gen = {task.name}
        next_gen = set()
        visited = set()
        while current_gen and time < time_window:
            cur_node = current_gen.pop()
            visited.add(cur_node)
            children = get_children(task_graph, cur_node)
            # next_gen.update({child.name for child in children if child.name not in visited})
            for child in children:
                if child.name not in visited:
                    visited.add(child.name)
                    time += child.cost / average_network_speed
                    if time >= time_window:
                        return count
                    count += 1
                    next_gen.add(child.name)
            if current_gen:
                continue
            else:   
                # visited.update(current_gen)
                current_gen = next_gen
                next_gen = set()
        return count
    effeciency_ranking = []
    # for task in task_graph.tasks:
    #     ability = compute_task_inspiring_ability(task)
    #     # counter += 1
    #     # heapq.heappush(effeciency_ranking, (-ability, counter, task.name))
    effeciency_ranks = {task.name: compute_task_inspiring_ability(task) for task in task_graph.tasks}
    return effeciency_ranks

        


def compute_inspiring_ability(task_graph: TaskGraph) -> dict[str, float]:
    """Compute the inspiring ability of tasks in a task graph.
    A task's inspiring ability is the total amount of tasks dependent on it.

    Args:
        task_graph (TaskGraph): The task graph to compute the inspiring ability for.

    Returns:
        ability_ranks: A dict of task_name to inspiring ability, where a task's inspiring ability is the total amount of tasks dependent on it.
        """
    
    topological_order = reversed(task_graph.topological_sort())
    reachable = {task: set() for task in task_graph.tasks}
    ability_ranking = []
    counter = 0
    for task in topological_order:
        for child in get_children(task_graph, task.name):
            reachable[task].add(child)
            reachable[task].update(reachable[child])
    ability_ranks = {task.name: (len(reachable[task])+1) for task in task_graph.tasks}
    return ability_ranks


def get_children(task_graph: TaskGraph, task_name: str) -> List[TaskGraphNode]:
    """Get the children of a task in a task graph.

    Args:
        task_graph (TaskGraph): The task graph to get the children from.
        task_name (str): The name of the task to get the children for.

    Returns:
        List[TaskGraphNode]: The child nodes of the task.
    """
    return [task_graph.get_task(edge.target)for edge in task_graph.out_edges(task_name) if edge.target not in {"__super_source__", "__super_sink__"}]




class InspriritScheduler(Scheduler):
    def __init__(self, scheduler: Scheduler, threshold: int, delta_ready: int, smoothing_rate: float = 0.8):
        super().__init__()
        self.scheduler = scheduler
        self.threshold = threshold
        self.delta_ready = delta_ready
        self.smoothing_rate = smoothing_rate

    def schedule(self, network, task_graph, schedule=None, min_start_time: float = 0.0):
        from saga.schedulers.online import InspiritController, InspiritEnvironment, TaskCompletionStep, ReadyChangeObserver
        env = InspiritEnvironment(
            network=network,
            task_graph=task_graph,
            scheduler=self.scheduler,
            step_strategy=TaskCompletionStep(),
            observer=ReadyChangeObserver(self.delta_ready),
            time_window=None,
            controller=InspiritController(smoothing_rate=self.smoothing_rate),
            on_step=None,
            dec_step=self.threshold,
            s_inc=self.threshold,
            s_dec=self.threshold,
        )
        return env.run()