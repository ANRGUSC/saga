import logging
import os
import pathlib
from typing import Set

from saga.schedulers import (
    BILScheduler,
    CpopScheduler,
    DuplexScheduler,
    ETFScheduler,
    FastestNodeScheduler,
    FCPScheduler,
    FLBScheduler,
    GDLScheduler,
    HeftScheduler,
    MaxMinScheduler,
    MCTScheduler,
    METScheduler,
    MinMinScheduler,
    OLBScheduler,
    WBAScheduler,
)

logging.basicConfig(level=logging.ERROR)

thisdir = pathlib.Path(__file__).parent.resolve()

saga_schedulers = {
    # Schedulers included in benchmarking results for the paper
    # "Comparing Task Graph Scheduling Algorithms: An Adversarial Approach"
    # https://arxiv.org/abs/2403.07120
    "BIL": BILScheduler(),
    "CPoP": CpopScheduler(),
    "Duplex": DuplexScheduler(),
    "ETF": ETFScheduler(),
    "FCP": FCPScheduler(),
    "FLB": FLBScheduler(),
    "FastestNode": FastestNodeScheduler(),
    "GDL": GDLScheduler(),
    "HEFT": HeftScheduler(),
    "MCT": MCTScheduler(),
    "MET": METScheduler(),
    "MaxMin": MaxMinScheduler(),
    "MinMin": MinMinScheduler(),
    "OLB": OLBScheduler(),
    "WBA": WBAScheduler(),
}

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
