import logging
import pathlib
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict
import numpy as np
import heapq

import networkx as nx

from saga import Network, Schedule, TaskGraph, Scheduler, TaskGraphNode
from saga.schedulers.online.environments import Environment, FrontierEnvironment
from saga.schedulers.online.controllers import FrontierPopController
from saga.schedulers.cpop import upward_rank
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



class FrontierHeftEnvironment(FrontierEnvironment):
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
            controller=FrontierPopController(
                insertion_strategy=GreedyInsert(
                    append_only=False,
                    compare=GreedyInsertCompareFuncs.EFT,
                    critical_path=False,
                    ),
            ),
            **kwargs
        )
        self.ready_condition: str = "p_committed"
        self.ready_node_only: bool = False
        self._bootstrap_insert = GreedyInsert(
            append_only=False,
            compare=GreedyInsertCompareFuncs.EFT,
            critical_path=False,
        )
        self.ranks:Dict[str, Tuple[(float, int)]] = self.compute_upward_ranks()
        self.priority_condition:Callable[[TaskGraphNode],float] = lambda x:self.ranks[x]

    def compute_upward_ranks(self):
        urank = upward_rank(self.network, self.task_graph)
        topological_sort = {
            node.name: i for i, node in enumerate(reversed(self.task_graph.topological_sort()))
        }
        return {node: (urank[node], topological_sort[node]) for node in urank}

class FrontierHeftScheduler(Scheduler):
    
    def schedule(self, network, task_graph):
        environment = FrontierHeftEnvironment(network,task_graph)
        return environment.run()