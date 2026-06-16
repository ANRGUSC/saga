from typing import Optional

from saga import Network, Schedule, TaskGraph, Scheduler
from saga.schedulers.online.environments import FrontierEnvironment
from saga.schedulers.online.controllers import FrontierPopController, InspiritController
from saga.schedulers.online.components import TaskEventStep, OnStepObserver, ReadyChangeObserver, CompositeObserver


class FIFOEnvironment(FrontierEnvironment):
    """First-in first-out scheduling."""

    def __init__(self, network: Network, task_graph: TaskGraph, **kwargs) -> None:
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
        return env.run()


SMOOTHING_RATE = 0.8


class InspiritFIFOScheduler(Scheduler):
    def __init__(self, threshold: int, delta_ready: int) -> None:
        super().__init__()
        self.threshold = threshold
        self.delta_ready = delta_ready

    def schedule(self, network, task_graph, schedule=None, min_start_time: float = 0.0):
        env = FrontierEnvironment(
            network=network,
            task_graph=task_graph,
            step_strategy=TaskEventStep(),
            observer=CompositeObserver([OnStepObserver(), ReadyChangeObserver(self.delta_ready)]),
            controller=InspiritController(
                smoothing_rate=SMOOTHING_RATE,
                dec_step=self.threshold,
                s_inc=self.threshold,
                s_dec=self.threshold,
                fill_controller=FrontierPopController(),
            ),
        )
        return env.run()
