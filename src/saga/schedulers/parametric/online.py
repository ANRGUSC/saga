from copy import deepcopy
import heapq
import numpy as np
from pydantic import BaseModel, Field
from enum import Enum
from typing import Callable, Tuple, Set, Optional

from saga import NetworkNode, Scheduler, ScheduledTask, TaskGraphNode
from saga.schedulers.parametric import IntialPriority, InsertTask, ParametricScheduler
from saga.schedulers.parametric.components import (
    insert_funcs, initial_priority_funcs
)
from saga.schedulers.heft import heft_rank_sort
from saga.schedulers.cpop import cpop_ranks
from saga.schedulers.stochastic import EstimateStochasticScheduler
from saga.stochastic import StochasticNetwork, StochasticScheduler, StochasticSchedule, StochasticScheduledTask, StochasticTaskGraph


from saga import Network, TaskGraph, Schedule, TaskGraphNode

import networkx as nx
from typing import Dict, Iterable, List, Hashable, Literal

from saga.utils.random_variable import RandomVariable


def get_next_task(schedule: Schedule, current_moment: float) -> ScheduledTask:
    """Returns the task with the smallest end time > current_moment
    
    Args:
        schedule: ...
        current_moment: ...

    Returns:
        ScheduledTask
    """
    return min(
        [
            task for _, tasks in schedule.mapping.items() for task in tasks
            if task.end > current_moment
        ],
        key=lambda task: task.end,
        default= None
)

def get_running_tasks(schedule: Schedule, current_moment: float) -> Set[ScheduledTask]:

    """
    Args: 
        schedule: ...
        current_moment ...

    Returns:
        set[ScheduledTask]
    """
    return {
        task for _, tasks in schedule.mapping.items() for task in tasks
        if (task.start < current_moment) and (task.end>current_moment)
    }


def get_finished_tasks(schedule: Schedule, current_moment: float) -> Set[ScheduledTask]:
    """Returns the set of all complete tasks up to the current moment
    Args: 
        schedule: ...
        current_moment ...

    Returns:
        set[ScheduledTask]
    """
    return {
        task for _, tasks in schedule.mapping.items() for task in tasks
        if (task.end <= current_moment)
    }

def create_partial_schedule(
        running_tasks:set[ScheduledTask], 
        finished_tasks:set[ScheduledTask], 
        schedule:Schedule,
        det_network: Network,
        det_task_graph: TaskGraph) -> Schedule:
    """ Returns a partial schedule with all commited tasks. 
        Running task endtimes are treated as their estimate values, 
        finished_tasks are treated with their actual value
    Args: 
        running_tasks: ...
        finished_tasks: ...
        schedule: ...
        det_network: ...
        det_task_graph: ...

    Returns:
        Schedule
    """
    partial_schedule:Schedule = Schedule(schedule.task_graph, schedule.network)

    for task in finished_tasks:
        partial_schedule.add_task(task)

    est_running_tasks:List[ScheduledTask] = deepcopy(running_tasks)

    for task in est_running_tasks:
        est_task_size = det_task_graph.get_task(task.name).cost
        est_network_speed = det_network.get_node(task.node).speed
        task.end = task.start + (est_task_size/est_network_speed)
        partial_schedule.add_task(task)

    return partial_schedule
    



class OnlineParametricScheduler(Scheduler):
    def __init__(self, 
                 scheduler: ParametricScheduler, 
                 estimate: Callable[[RandomVariable], float], 
                 seed: Optional[int]= None
                 )->None:
        super().__init__()
        self.seed = seed
        self.stochastic_scheduler = EstimateStochasticScheduler(
            scheduler=scheduler,
            estimate=estimate
        )

       
    def schedule_iterative(self,
                           network: StochasticNetwork,
                           task_graph: StochasticTaskGraph) -> Tuple[List[Schedule], List[Schedule], List[Schedule]]:
        """Online scheduling algorithm that produces a schedule, then waits for
        the next task to finish before producing the next schedule.

        Args:
            network (nx.Graph): The network graph where tasks will be scheduled.
            task_graph (nx.DiGraph): The directed task graph containing tasks and their dependencies.

        Returns:
            Tuple[List[Schedule], List[Schedule], List[Schedule]:
                A tuple containing three lists:
                - schedules_actual: The actual schedules after all tasks have been scheduled.
                - schedules_estimate: The estimated schedules based on the parametric scheduler and estimate method.
                - schedules_partial: The partial schedules including finished tasks and started tasks with estimate endtimes.
                
        """
        #tracking
        schedules_estimate: List[Schedule] = [] 
        schedules_actual: List[Schedule] = []
        schedules_partial: List[Schedule] = []
        remaining_tasks: set[StochasticScheduledTask] = {task for task in task_graph}

        #generating our initial estimate schedule and our estimations for network and task graph values
        initial_estimate_schedule, det_network, det_task_graph = self.stochastic_scheduler.schedule(network=network,task_graph=task_graph)
        
        
        estimate_schedule: StochasticSchedule = initial_estimate_schedule
        schedules_estimate.append(estimate_schedule)
        current_moment = 0.0
        actual_task_graph:TaskGraph = task_graph.sample()
        actual_network:Network = network.sample()
        schedule_actual:Schedule = Schedule(network=actual_network, task_graph=actual_task_graph, mapping=None)
        
        
        while remaining_tasks: 
            """
            while remaining tasks: 
            1. Determinize values from our current estimate schedule.
            2. Set our running tasks and finished tasks based on the current moment. Update remaining tasks.
            3. Obtain partial schedule. This is the schedule containing all finished tasks, and all running tasks locked.
            4. Generate a new estimate schedule, scheduling around the partial schedule
            5. Find the next task to finish. If none remaining, return
            6. Advance current moment to the time of the next task to finish
            """
            #------step 1------
            schedule_actual:Schedule = estimate_schedule.determinize(actual_network, actual_task_graph)
            schedules_actual.append(schedule_actual)

            #------step 2------
            running_tasks: set[ScheduledTask] = get_running_tasks(schedule_actual, current_moment)
            finished_tasks: set[ScheduledTask] = get_finished_tasks(schedule_actual, current_moment)
            remaining_tasks -= running_tasks | finished_tasks

            #------step 3------
            partial_schedule:Schedule = create_partial_schedule(
                running_tasks, 
                finished_tasks, 
                schedule_actual,
                det_network,
                det_task_graph
                )
            schedules_partial.append(partial_schedule)

            #------step 4------
            estimate_schedule:StochasticSchedule = self.stochastic_scheduler.schedule(
                network=network,
                task_graph=task_graph,
                schedule=partial_schedule,
                min_start_time=current_moment
                )[0]
            
            schedules_estimate.append(estimate_schedule)
            #------step 5------
            next_task:ScheduledTask|None = get_next_task(schedule_actual, current_moment)
            if next_task == None:
                return schedules_actual, schedules_estimate, schedules_partial

            #------step 6------
            current_moment = next_task.end
        
        
        

    def schedule(self,
                 network: StochasticNetwork,
                 task_graph: StochasticTaskGraph) -> Schedule:
        """Online scheduling algorithm that produces a schedule, then waits for
        the next task to finish before producing the next schedule. This method
        returns only the final schedule after all tasks have been scheduled.

        Args:
            network (nx.Graph): The network graph where tasks will be scheduled.
            task_graph (nx.DiGraph): The directed task graph containing tasks and their dependencies.
        
        Returns:
            Dict[Hashable, List[Task]]: A dictionary mapping nodes to lists of tasks scheduled on those nodes.
        """
        schedules, _, _ = self.schedule_iterative(network, task_graph)
        # Return the last schedule in the list
        return schedules[-1]

schedulers: Dict[str, ParametricScheduler] = {}
for name, insert_func in insert_funcs.items():
    for intial_priority_name, initial_priority_func in initial_priority_funcs.items():
        schedulers[f"{name}_{intial_priority_name}"] = ParametricScheduler(
            initial_priority=initial_priority_func,
            insert_task=insert_func
        )

