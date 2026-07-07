from saga.schedulers.stochastic.estimate_stochastic_scheduler import EstimateStochasticScheduler
from saga.schedulers.heft import HeftScheduler


class SheftScheduler(EstimateStochasticScheduler):
    def __init__(self) -> None:
        super().__init__(
            scheduler=HeftScheduler(), estimate=lambda rv: rv.mean() + rv.std()
        )
