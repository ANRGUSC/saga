import copy
import logging  # pylint: disable=missing-module-docstring
import pathlib
from typing import Optional

import dill as pickle
from saga.schedulers import (BILScheduler, CpopScheduler, DuplexScheduler,
                             ETFScheduler, FastestNodeScheduler, FCPScheduler,
                             FLBScheduler, GDLScheduler, HeftScheduler,
                             MaxMinScheduler, MCTScheduler, METScheduler,
                             MinMinScheduler, OLBScheduler, WBAScheduler)
from saga.utils.random_graphs import (add_random_weights, get_chain_dag,
                                      get_network)

from changes import (NetworkChangeEdgeWeight, NetworkChangeNodeWeight,
                     TaskGraphAddDependency, TaskGraphChangeDependencyWeight,
                     TaskGraphChangeTaskWeight, TaskGraphDeleteDependency)
from simulated_annealing import SimulatedAnnealing

thisdir = pathlib.Path(__file__).parent

logging.basicConfig(level=logging.INFO)

# set logging format [time] [level] message
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s')

def main():
    """Run the experiments."""
    schedulers = {
        "CPOP": CpopScheduler(),
        "Duplex": DuplexScheduler(),
        "ETF": ETFScheduler(),
        "HEFT": HeftScheduler(),
        "Fastest Node": FastestNodeScheduler(),
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
    }
    base_schedulers = copy.deepcopy(schedulers)
    # algorithms that only work when the compute speed is the same for all nodes
    homogenous_comp_algs = {"ETF", "FCP", "FLB"}
    # algorithms that only work when the communication speed is the same for all network edges
    homogenous_comm_algs = {"BIL", "GDL", "FCP", "FLB"}
    rerun_for = ["FCP"]

    # Configuration Parameters
    max_iterations = [1000]*5
    max_temp = 10
    min_temp = 0.1
    cooling_rate = 0.99
    get_makespan = SimulatedAnnealing.DEFAULT_GET_MAKESPAN
    skip_existing = True
    task_graph_changes = [
        TaskGraphAddDependency, TaskGraphDeleteDependency,
        TaskGraphChangeDependencyWeight, TaskGraphChangeTaskWeight
    ]

    for base_scheduler_name, base_scheduler in base_schedulers.items():
        for scheduler_name, scheduler in schedulers.items():
            savepath = thisdir / "results" / base_scheduler_name / f"{scheduler_name}.pkl"
            if savepath.exists() and skip_existing and not {base_scheduler_name, scheduler_name}.intersection(rerun_for):
                logging.info("Skipping experiment for %s/%s", scheduler_name, base_scheduler_name)
                continue

            best_run: Optional[SimulatedAnnealing] = None
            try:
                for max_iter in max_iterations:
                    if scheduler_name == base_scheduler_name:
                        continue
                    # Brute force is optimal, so no point trying to find a worst-case scenario
                    if scheduler_name == "Brute Force":
                        continue

                    logging.info(
                        "Running experiment for %s/%s with %s iterations",
                        scheduler_name, base_scheduler_name, max_iter
                    )
                    network = add_random_weights(get_network())
                    task_graph = add_random_weights(get_chain_dag())

                    change_types = copy.deepcopy(task_graph_changes)
                    if scheduler_name in homogenous_comp_algs or base_scheduler_name in homogenous_comp_algs:
                        for node in network.nodes:
                            network.nodes[node]["weight"] = 1
                    else:
                        change_types.append(NetworkChangeNodeWeight)

                    if scheduler_name in homogenous_comm_algs or base_scheduler_name in homogenous_comm_algs:
                        for edge in network.edges:
                            if edge[0] == edge[1]:
                                network.edges[edge]["weight"] = 1e9
                            else:
                                network.edges[edge]["weight"] = 1
                    else:
                        change_types.append(NetworkChangeEdgeWeight)


                    simulated_annealing = SimulatedAnnealing(
                        task_graph=task_graph,
                        network=network,
                        scheduler=scheduler,
                        base_scheduler=base_scheduler,
                        min_temp=min_temp,
                        cooling_rate=cooling_rate,
                        max_temp=max_temp,
                        max_iterations=max_iter,
                        change_types=change_types,
                        get_makespan=get_makespan
                    )

                    simulated_annealing.run()

                    if best_run is None or (best_run.iterations[-1].best_energy <
                                            simulated_annealing.iterations[-1].best_energy):
                        best_run = simulated_annealing
                        savepath = thisdir.joinpath(
                            "results", base_scheduler_name, f"{scheduler_name}.pkl"
                        )
                        savepath.parent.mkdir(parents=True, exist_ok=True)
                        savepath.write_bytes(pickle.dumps(simulated_annealing))
            except Exception as exp:
                logging.error(
                    "Error running experiment for %s/%s with %s iterations",
                    scheduler_name, base_scheduler_name, max_iter
                )
                logging.error(exp)
                raise exp


if __name__ == "__main__":
    main()
