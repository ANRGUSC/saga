import logging  # pylint: disable=missing-module-docstring
import pathlib
from itertools import product

from saga.schedulers import (BILScheduler, CpopScheduler, DuplexScheduler,
                             ETFScheduler, FastestNodeScheduler, FCPScheduler,
                             FLBScheduler, GDLScheduler, HeftScheduler,
                             MaxMinScheduler, MCTScheduler, METScheduler,
                             MinMinScheduler, OLBScheduler, WBAScheduler, SDBATSScheduler)
from saga.pisa import run_experiments

thisdir = pathlib.Path(__file__).parent

logging.basicConfig(level=logging.INFO)

# set logging format [time] [level] message
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s')


# algorithms that only work when the compute speed is the same for all nodes
homogenous_comp_algs = {"ETF", "FCP", "FLB"}
# algorithms that only work when the communication speed is the same for all network edges
homogenous_comm_algs = {"BIL", "GDL", "FCP", "FLB"}
rerun_schedulers = ["FLB"]
rerun_base_schedulers = []


SCHEDULERS = {
    "CPOP": CpopScheduler(),
    "HEFT": HeftScheduler(),
    "Duplex": DuplexScheduler(),
    "ETF": ETFScheduler(),
    "FastestNode": FastestNodeScheduler(),
    "FCP": FCPScheduler(),
    "GDL": GDLScheduler(),
    "MaxMin": MaxMinScheduler(),
    "MinMin": MinMinScheduler(),
    "MCT": MCTScheduler(),
    "MET": METScheduler(),
    "OLB": OLBScheduler(),
    "BIL": BILScheduler(),
    "WBA": WBAScheduler(),
    "FLB": FLBScheduler(),
    "SDBATS": SDBATSScheduler()
}
def run(output_path: pathlib.Path): # pylint: disable=too-many-locals, too-many-statements
    """Run first set of experiments."""
    scheduler_pairs = list(product(SCHEDULERS.items(), SCHEDULERS.items()))
    run_experiments(
        scheduler_pairs=scheduler_pairs,
        max_iterations=1000,
        num_tries=10,
        max_temp=10,
        min_temp=0.1,
        cooling_rate=0.99,
        skip_existing=True,
        output_path=output_path
    )


def main():
    logging.basicConfig(level=logging.INFO)

    resultsdir = thisdir.joinpath("results")

    run(resultsdir)

if __name__ == "__main__":
    main()