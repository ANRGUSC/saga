from saga.schedulers.heft import HeftScheduler
from saga.schedulers.stochastic.determinizer import Determinizer


class MeanHeftScheduler(Determinizer):
    def __init__(self) -> None:
        super().__init__(scheduler=HeftScheduler(), determinize=lambda rv: rv.mean())
