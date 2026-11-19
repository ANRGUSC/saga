"""Shared configuration for the priority-function experiment.

Every scheduler here is a ParametricScheduler, so the only thing that differs between a
baseline and its randomized counterpart is the initial priority (the task order). The
insertion policy is held fixed: EFT insertion for HEFT, EFT insertion with critical-path
pinning for CPoP.
"""

import logging
import os
import pathlib
from typing import Callable, Dict, Set

from saga import Scheduler
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    CPoPRanking,
    RandomInsert,
    RandomTopological,
    UpwardRanking,
    insert_funcs,
)

logging.basicConfig(level=logging.ERROR)

thisdir = pathlib.Path(__file__).parent.resolve()

SchedulerFactory = Callable[[int], Scheduler]
"""Builds a scheduler for a given sample seed. Deterministic schedulers ignore the seed."""


def heft(seed: int) -> Scheduler:
    return ParametricScheduler(
        initial_priority=UpwardRanking(), insert_task=insert_funcs["EFT_Insert"]
    )


def heft_random_order(seed: int) -> Scheduler:
    return ParametricScheduler(
        initial_priority=RandomTopological(seed=seed),
        insert_task=insert_funcs["EFT_Insert"],
    )


def cpop(seed: int) -> Scheduler:
    return ParametricScheduler(
        initial_priority=CPoPRanking(), insert_task=insert_funcs["EFT_Insert_CP"]
    )


def cpop_random_order(seed: int) -> Scheduler:
    return ParametricScheduler(
        initial_priority=RandomTopological(seed=seed),
        insert_task=insert_funcs["EFT_Insert_CP"],
    )


def random_order_random_node(seed: int) -> Scheduler:
    return ParametricScheduler(
        initial_priority=RandomTopological(seed=seed),
        insert_task=RandomInsert(seed=seed, append_only=False),
    )


saga_schedulers: Dict[str, SchedulerFactory] = {
    "HEFT": heft,
    "HEFT_RandomOrder": heft_random_order,
    "CPoP": cpop,
    "CPoP_RandomOrder": cpop_random_order,
    "RandomOrder_RandomNode": random_order_random_node,
}

# Number of independent samples (seeds) per instance. Deterministic schedulers need one.
num_samples: Dict[str, int] = {
    "HEFT": 1,
    "HEFT_RandomOrder": 100,
    "CPoP": 1,
    "CPoP_RandomOrder": 100,
    "RandomOrder_RandomNode": 100,
}

# Every instance's network is scaled so that the mean communication time over the mean
# computation time equals this value. Without it the riotbench fog networks have CCR in the
# hundreds and every list scheduler collapses onto the single cloud node.
target_ccr: float = 1.0

exclude_datasets: Set[str] = {
    "bwa",
    "cycles",
    "epigenomics",
    "genome",
    "montage",
    "seismology",
    "soykb",
    "srasearch",
    "blast",
}

datadir = thisdir.joinpath("data", "benchmarking")
resultsdir = thisdir.joinpath("results", "benchmarking")
outputdir = thisdir.joinpath("output", "benchmarking")

num_processors = max(1, (os.cpu_count() or 1) - 3)

os.environ["SAGA_DATA_DIR"] = str(thisdir.joinpath("data", "benchmarking"))
