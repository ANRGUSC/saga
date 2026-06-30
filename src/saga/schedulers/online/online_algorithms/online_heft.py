import logging
import pathlib
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, cast
import numpy as np

import networkx as nx

from saga import Network, Schedule, TaskGraph, Scheduler
from saga.schedulers.online.environments import StochasticEnvironment
from saga.stochastic import StochasticNetwork, StochasticScheduler, StochasticSchedule, StochasticScheduledTask, StochasticTaskGraph
from saga.schedulers.online import (
    Environment,
    ReschedulePolicy,
)
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    GreedyInsert,
    GreedyInsertCompareFuncs,
    UpwardRanking,
)
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

class OnlineHEFTEnvironment(StochasticEnvironment):
    def __init__(
        self,
        network: StochasticNetwork,
        task_graph: StochasticTaskGraph,
        on_step: Optional[Callable[["Environment"], None]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            network=network,
            task_graph=task_graph,
            scheduler=ParametricScheduler(
                initial_priority=UpwardRanking(),
                insert_task=GreedyInsert(
                    append_only=False,
                    compare=GreedyInsertCompareFuncs.EFT,
                    critical_path=False,
                ),
            ),
            estimate=lambda rv: rv.mean(),
            policy=ReschedulePolicy(),
            on_step=on_step,
            seed=seed,
        )

class OnlineHEFT(Scheduler):
    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        env = OnlineHEFTEnvironment(
            cast(StochasticNetwork, network),
            cast(StochasticTaskGraph, task_graph),
        )
        return env.run()
    