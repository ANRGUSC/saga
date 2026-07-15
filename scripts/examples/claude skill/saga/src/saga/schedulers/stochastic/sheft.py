from saga.schedulers.stochastic.determinizer import Determinizer
from saga.schedulers.heft import HeftScheduler


class SheftScheduler(Determinizer):
    def __init__(self) -> None:
        super().__init__(
            scheduler=HeftScheduler(), determinize=lambda rv: rv.mean() + rv.std()
        )
