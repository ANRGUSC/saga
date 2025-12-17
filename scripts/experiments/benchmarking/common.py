import os
import pathlib
import logging

from saga.schedulers import (
    BILScheduler, CpopScheduler, DuplexScheduler, ETFScheduler, FCPScheduler,
    FLBScheduler, FastestNodeScheduler, GDLScheduler, HeftScheduler,
    MCTScheduler, METScheduler, MaxMinScheduler, MinMinScheduler,
    OLBScheduler, WBAScheduler
)


logging.basicConfig(level=logging.INFO)

thisdir = pathlib.Path(__file__).parent.resolve()

exclude_schedulers = []
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
    "WBA": WBAScheduler()
}

datadir = thisdir.joinpath("data", "benchmarking")
resultsdir = thisdir.joinpath("results", "benchmarking")
outputdir = thisdir.joinpath("output", "benchmarking")

os.environ["SAGA_DATA_DIR"] = str(thisdir.joinpath("data", "benchmarking"))
