from typing import Any, Optional

from saga import Network, Schedule, TaskGraph, Scheduler
from saga.schedulers.online.environments import FrontierEnvironment
from saga.schedulers.online.policies import FrontierFillPolicy, InspiritPolicy


class FIFOEnvironment(FrontierEnvironment):
    """First-in first-out scheduling.

    Dispatches the oldest-ready task to fill available nodes on every step (the
    frontier is a min-heap keyed on arrival time).
    """

    def __init__(self, network: Network, task_graph: TaskGraph, **kwargs: Any) -> None:
        super().__init__(
            network=network,
            task_graph=task_graph,
            policy=FrontierFillPolicy(),
            **kwargs
        )
        self.ready_condition: str = "p_complete"
        self.ready_node_only: bool = True


class FIFOScheduler(Scheduler):
    """Online FIFO scheduler that builds the schedule incrementally as tasks finish."""

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        env = FIFOEnvironment(network, task_graph)
        return env.run()


SMOOTHING_RATE = 0.8


class InspiritFIFOScheduler(Scheduler):
    """FIFO scheduler with an Inspirit policy layered on to maintain a ready-task pool."""

    def __init__(self, threshold: int, delta_ready: int) -> None:
        super().__init__()
        self.threshold = threshold
        self.delta_ready = delta_ready

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        env = FrontierEnvironment(
            network=network,
            task_graph=task_graph,
            policy=InspiritPolicy(
                smoothing_rate=SMOOTHING_RATE,
                delta_ready=self.delta_ready,
                dec_step=self.threshold,
                s_inc=self.threshold,
                s_dec=self.threshold,
                fill_policy=FrontierFillPolicy(),
            ),
        )
        return env.run()
