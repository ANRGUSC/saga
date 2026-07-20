from typing import Callable, Optional, cast


from saga import Network, Schedule, TaskGraph, Scheduler
from saga.schedulers.online.environment import Environment, StochasticEnvironment
from saga.stochastic import StochasticNetwork, StochasticTaskGraph
from saga.schedulers.online.policy import ReschedulePolicy
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    GreedyInsert,
    GreedyInsertCompareFuncs,
    UpwardRanking,
)


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
