"""Deterministic RIoTBench loaders.

RIoTBench has no real traces, so the generative model lives in
saga.schedulers.stochastic.data.riotbench as random variables (topology, per-operator
peak-rate costs, selectivity edge sizes, and the modeled lognormal noise). A deterministic
instance is one realized draw of that model: we build the stochastic version with a single
sample per variable and collapse it with .sample(). Variety across instances comes from the
noise, and each instance still exposes exact scalar costs (perfect information).

This inverts the WfCommons layering (there the deterministic trace loader is the base and
the stochastic layer fits distributions on top), because for RIoTBench the random-variable
form is the fuller specification and the scalar is the degenerate collapse.
"""
from functools import partial
from typing import Callable, List

from saga import Network, TaskGraph
from saga.schedulers.stochastic.data import riotbench as _stochastic
from saga.schedulers.stochastic.data.riotbench import gaussian  # re-exported for callers


def get_fog_networks(num: int, **kwargs) -> List[Network]:
    """Tiered edge/fog/cloud networks; see the stochastic loader for parameters."""
    return [net.sample() for net in _stochastic.get_fog_networks(num, num_samples=1, **kwargs)]


def get_etl_task_graphs(
    num: int,
    get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
    count_window: int = 10,
    batch_window: int = 10,
) -> List[TaskGraph]:
    """ETL (Extract-Transform-Load) dataflows, RIoTBench Fig. 3a; see stochastic loader."""
    return [
        tg.sample()
        for tg in _stochastic.get_etl_task_graphs(
            num,
            get_input_size=get_input_size,
            count_window=count_window,
            batch_window=batch_window,
            num_samples=1,
        )
    ]


def get_stats_task_graphs(
    num: int,
    get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
    count_window: int = 10,
    plot_window: int = 10,
) -> List[TaskGraph]:
    """STATS (Statistical Summarization) dataflows, RIoTBench Fig. 3b; see stochastic loader."""
    return [
        tg.sample()
        for tg in _stochastic.get_stats_task_graphs(
            num,
            get_input_size=get_input_size,
            count_window=count_window,
            plot_window=plot_window,
            num_samples=1,
        )
    ]


def get_train_task_graphs(
    num: int,
    get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
    train_window: int = 1000,
) -> List[TaskGraph]:
    """TRAIN (Model Training) dataflows, RIoTBench Fig. 3c; see stochastic loader."""
    return [
        tg.sample()
        for tg in _stochastic.get_train_task_graphs(
            num,
            get_input_size=get_input_size,
            train_window=train_window,
            num_samples=1,
        )
    ]


def get_predict_task_graphs(
    num: int,
    get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
    count_window: int = 10,
) -> List[TaskGraph]:
    """PRED (Predictive Analytics) dataflows, RIoTBench Fig. 3d; see stochastic loader."""
    return [
        tg.sample()
        for tg in _stochastic.get_predict_task_graphs(
            num,
            get_input_size=get_input_size,
            count_window=count_window,
            num_samples=1,
        )
    ]
