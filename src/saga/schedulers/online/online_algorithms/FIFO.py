import logging
import pathlib
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
import heapq

import networkx as nx

from saga import Network, Schedule, TaskGraph, Scheduler
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
    ) -> None:
        super().__init__(
            network=network,
            task_graph=task_graph,
            step_strategy=TaskEventStep(),
            observer=OnStepObserver(),
            controller=FrontierPopController(),
        )
    def _update_task_state(self) -> None:
        super()._update_task_state()

        finished_names = {t.name for t in self.finished_tasks}
        committed_names = finished_names | {t.name for t in self.running_tasks}
        scheduled_names = {
            t.name for tasks in self.schedule.mapping.values() for t in tasks
        }

        # Drop committed tasks from the frontier
        stale = committed_names & self.frontier_set
        if stale:
            self.frontier_set -= stale
            self.frontier = [(t, n) for t, n in self.frontier if n not in stale]
            heapq.heapify(self.frontier)

        # Enqueue unscheduled tasks whose predecessors have all finished
        for task in self.task_graph.tasks:
            name = task.name
            if name in scheduled_names or name in self.frontier_set:
                continue
            predecessors = {dep.source for dep in self.task_graph.in_edges(name)}
            if predecessors.issubset(finished_names):
                self.frontier_set.add(name)
                heapq.heappush(self.frontier, (self.current_time, name))


class FIFO(Scheduler):
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
        env.reset(schedule=schedule, min_start_time=min_start_time)
        # Drain tasks immediately ready at min_start_time before entering the
        # event-driven loop. TaskEventStep requires a future event in the
        # schedule to advance; if the partial schedule only has past-committed
        # tasks there are no such events yet, so the controller must fire once
        # per ready frontier task before the first step can proceed.
        while env.frontier:
            trigger = env.observer.observe(env)
            if trigger is not None and env.controller is not None:
                env.schedule = env.controller.control(env, trigger)
            env._update_task_state()
            env._update_network_state()
        while env.step():
            pass
        return env.schedule
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
            scheduler=FIFO(),
            step_strategy=TaskEventStep(),
            observer=ReadyChangeObserver(delta_ready),
            time_window=None,
            controller=InspiritController(smoothing_rate=SMOOTHING_RATE),
            on_step=on_step,
            dec_step=dec_step,
            s_inc=s_inc,
            s_dec=s_dec,
        )



class Inspirit_FIFO(Scheduler):
    def __init__(self, network, task_graph, environment: Inspirit_FIFO_Environment):
        super().__init__()
        self.environment = environment
    def schedule(self,
            network,
            task_graph,
            schedule: Optional[Schedule] = None,
            min_start_time: float = 0.0,
        ):
        return self.environment.run()




