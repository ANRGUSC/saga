from typing import Callable, Optional


from saga import Schedule
from saga.schedulers.online.environment import Environment, StochasticEnvironment
from saga.stochastic import OnlineScheduler, StochasticNetwork, StochasticTaskGraph
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


class OnlineHEFT(OnlineScheduler):
    def schedule(
        self,
        network: StochasticNetwork,
        task_graph: StochasticTaskGraph,
    ) -> Schedule:
        env = OnlineHEFTEnvironment(network, task_graph)
        return env.run()
