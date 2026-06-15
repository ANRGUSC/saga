import logging
import pathlib
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
import heapq

import networkx as nx

from saga import Network, Schedule, TaskGraph, Scheduler, TaskGraphNode
from saga.schedulers.online.environments import Environment, FrontierEnvironment
from saga.schedulers.online.controllers import FrontierPopController
from saga.schedulers.online import (
    Environment,
    InspiritController,
    InspiritEnvironment,
    ReadyChangeObserver,
    OnStepObserver,
    TaskCompletionStep,
    RescheduleController,
    StepStrategy,
    Observer,
    OnStepObserver,
    TaskEventStep
)
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    GreedyInsert,
    GreedyInsertCompareFuncs,
    UpwardRanking,
)
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

class FIFOEnvironment(FrontierEnvironment):
    """First in first out scheduling"""

    def __init__(
        self,
        network: Network,
        task_graph: TaskGraph,
        **kwargs
    ) -> None:
        super().__init__(
            network=network,
            task_graph=task_graph,
            step_strategy=TaskEventStep(),
            observer=OnStepObserver(),
            controller=FrontierPopController(),
            **kwargs
        )
        self.ready_condition: str = "p_complete"
        self.ready_node_only: bool = True

    

class FIFOScheduler(Scheduler):
    def __init__(self):
        super().__init__()
    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        env = FIFOEnvironment(network, task_graph)
        # env.reset(schedule=schedule, min_start_time=min_start_time)
        return env.run()
        # while env.frontier:
        #     trigger = env.observer.observe(env)
        #     if trigger is not None and env.controller is not None:
        #         env.schedule = env.controller.control(env, trigger)
        #     env._update_task_state(lambda _: env.current_time)
        #     env._update_network_state()
        # while env.step(): #this also calls update_task_state
        #     pass
        # return env.schedule
SMOOTHING_RATE = 0.8  
class Inspirit_FIFO_Environment(InspiritEnvironment):
    def __init__(
            self,
            network,
            task_graph,
            delta_ready,
            on_step,
            dec_step,
            s_inc,
            s_dec):
        
        super().__init__(
            network=network,
            task_graph=task_graph,
            scheduler=FIFOScheduler(),
            step_strategy=TaskEventStep(),
            observer=ReadyChangeObserver(delta_ready),
            time_window=None,
            controller=InspiritController(smoothing_rate=SMOOTHING_RATE),
            on_step=on_step,
            dec_step=dec_step,
            s_inc=s_inc,
            s_dec=s_dec,
        )



class InspiritFIFOScheduler(Scheduler):
    def __init__(self, threshold: int, delta_ready: int) -> None:
        super().__init__()
        self.threshold = threshold
        self.delta_ready = delta_ready

    def schedule(self, network, task_graph, schedule=None, min_start_time: float = 0.0):
        env = Inspirit_FIFO_Environment(
            network=network,
            task_graph=task_graph,
            on_step=None,
            delta_ready=self.delta_ready,
            dec_step=self.threshold,
            s_inc=self.threshold,
            s_dec=self.threshold,
        )
        return env.run()




