from saga.schedulers.heft import HeftScheduler
from saga.schedulers.stochastic.estimate_stochastic_scheduler import EstimateStochasticScheduler


class MeanHeftScheduler(EstimateStochasticScheduler):
    def __init__(self) -> None:
        super().__init__(scheduler=HeftScheduler(), estimate=lambda rv: rv.mean())
