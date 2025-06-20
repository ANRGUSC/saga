from functools import partial
from typing import Dict, Hashable, List, Tuple
from copy import deepcopy
from itertools import product
import os
import time
import pathlib
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from saga.utils.draw import gradient_heatmap
from multiprocessing import Pool, Value, Lock, cpu_count

from .parametric import TempHeftScheduler
from saga.scheduler import Scheduler, Task
from saga.utils.online_tools import schedule_estimate_to_actual, get_offline_instance





class OnlineParametricScheduler(Scheduler):
    def __init__(self, param):
        self.heft_scheduler = TempHeftScheduler()

    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """
        A 'live' scheduling loop using HEFT in a dynamic manner.
        We assume each node/task has 'weight_actual' and 'weight_estimate' attributes.
        
        1) No longer needed
        2) Calls the standard HEFT for an 'estimated' schedule.
        3) Converts that schedule to 'actual' times using schedule_estimate_to_actual.
        4) Commits the earliest-finishing new task to schedule_actual.
        5) Repeats until all tasks are scheduled.        

        comp_schedule = {
        "Node1": [Task(node="Node1", name="TaskA", start=0, end=5)],
        "Node2": [Task(node="Node2", name="TaskB", start=3, end=8)],
        """
        schedule_actual: Dict[Hashable, List[Task]] = {
            node: [] for node in network.nodes
        }
        tasks_actual: Dict[str, Task] = {}
        current_time = 0
        iteration = 0

        #need to add another if statement that schecks if it is sufferage or not 
        
        while len(tasks_actual) < len(task_graph.nodes):
            schedule_actual_hypothetical = schedule_estimate_to_actual(
                network,
                task_graph,
                self.heft_scheduler.schedule(network, task_graph, schedule_actual, min_start_time=current_time) 
            )

            tasks: List[Task] = sorted([task for node_tasks in schedule_actual_hypothetical.values() for task in node_tasks], key=lambda x: x.start)
            next_task: Task = min(
                [task for task in tasks if task.name not in tasks_actual],
                key=lambda x: x.end
            )
            #move the time up
            current_time = next_task.end

            #add the next task to schedule_actual
            #if statement is just a failsafe to make sure we are not scheduling any hazardous task            
            for task in tasks:
                if task.start <= current_time and task.name not in tasks_actual:
                    schedule_actual[task.node].append(task)
                    tasks_actual[task.name] = task
            

        return schedule_actual
