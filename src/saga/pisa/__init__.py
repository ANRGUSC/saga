import copy
import logging  # pylint: disable=missing-module-docstring
import pathlib
import random
from typing import List, Optional, Tuple

import dill as pickle
from changes import (NetworkChangeEdgeWeight, NetworkChangeNodeWeight,
                     TaskGraphAddDependency, TaskGraphChangeDependencyWeight,
                     TaskGraphChangeTaskWeight, TaskGraphDeleteDependency)

from saga.scheduler import Scheduler
from saga.utils.random_graphs import (add_random_weights, get_chain_dag,
                                      get_network)
from saga.pisa.simulated_annealing import SimulatedAnnealing

thisdir = pathlib.Path(__file__).parent


logging.basicConfig(level=logging.INFO)

# set logging format [time] [level] message
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s')


def run_experiments(scheduler_pairs: List[Tuple[Tuple[str, Scheduler], Tuple[str, Scheduler]]],
                    max_iterations: int,
                    num_tries: int,
                    max_temp: float,
                    min_temp: float,
                    cooling_rate: float,
                    skip_existing: bool = True,
                    output_path: pathlib.Path = thisdir.joinpath("results"),
                    node_range: Tuple[int, int] = (3, 5),
                    task_range: Tuple[int, int] = (3, 5),
                    homogenous_comp_algs: set = None,
                    homogenous_comm_algs: set = None,
                    rerun_schedulers: list = None,
                    rerun_base_schedulers: list = None) -> None:
    """Run the experiments.
    
    
    Args:
        scheduler_pairs (List[Tuple[Tuple[str, Scheduler], Tuple[str, Scheduler]]]): list of tuples of (scheduler_name, scheduler) pairs.
            The first scheduler is the scheduler to be tested, and the second scheduler is the base scheduler.
        max_iterations (int): maximum number of iterations to run each experiment for.
        num_tries (int): number of times to run each experiment.
        max_temp (float): maximum temperature for simulated annealing.
        min_temp (float): minimum temperature for simulated annealing.
        cooling_rate (float): cooling rate for simulated annealing.
        skip_existing (bool, optional): whether to skip existing experiments. Defaults to True.
        output_path (pathlib.Path, optional): path to save results. Defaults to thisdir.joinpath("results").
        node_range (Tuple[int, int], optional): range of nodes for random network. Defaults to (3, 5).
        task_range (Tuple[int, int], optional): range of tasks for random task graph. Defaults to (3, 5).
        homogenous_comp_algs (set, optional): algorithms that only work when the compute speed is the same for all nodes. Defaults to None.
        homogenous_comm_algs (set, optional): algorithms that only work when the communication speed is the same for all network edges. Defaults to None.
        rerun_schedulers (list, optional): list of scheduler names to rerun. Defaults to None.
        rerun_base_schedulers (list, optional): list of base scheduler names to rerun. Defaults to None.
        
    Raises:
        exp: any exception raised during experiment.
    """

    if homogenous_comp_algs is None:
        homogenous_comp_algs = set()
    if homogenous_comm_algs is None:
        homogenous_comm_algs = set()
    if rerun_schedulers is None:
        rerun_schedulers = []
    if rerun_base_schedulers is None:
        rerun_base_schedulers = []

    task_graph_changes = [
        TaskGraphAddDependency, TaskGraphDeleteDependency,
        TaskGraphChangeDependencyWeight, TaskGraphChangeTaskWeight
    ]
    get_makespan = SimulatedAnnealing.default_get_makespan

    all_iterations = [max_iterations]*num_tries
    for (scheduler_name, scheduler), (base_scheduler_name, base_scheduler) in scheduler_pairs:
    # for base_scheduler_name, base_scheduler in base_schedulers.items(): # pylint: disable=too-many-nested-blocks
    #     for scheduler_name, scheduler in schedulers.items():
        savepath = output_path / base_scheduler_name / f"{scheduler_name}.pkl"
        if (savepath.exists() and skip_existing
            and not scheduler_name in rerun_schedulers
            and not base_scheduler_name in rerun_base_schedulers):
            logging.info("Skipping experiment for %s/%s", scheduler_name, base_scheduler_name)
            continue

        best_run: Optional[SimulatedAnnealing] = None
        try:
            for max_iter in all_iterations:
                if scheduler_name == base_scheduler_name:
                    continue
                # Brute force is optimal, so no point trying to find a worst-case scenario
                if scheduler_name == "Brute Force":
                    continue

                logging.info(
                    "Running experiment for %s/%s with %s iterations",
                    scheduler_name, base_scheduler_name, max_iter
                )

                network = add_random_weights(get_network(random.randint(*node_range)))
                task_graph = add_random_weights(get_chain_dag(random.randint(*task_range)))

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
                    savepath = output_path.joinpath(base_scheduler_name, f"{scheduler_name}.pkl")
                    savepath.parent.mkdir(parents=True, exist_ok=True)
                    savepath.write_bytes(pickle.dumps(simulated_annealing))
        except Exception as exp:
            logging.error(
                "Error running experiment for %s/%s with %s iterations",
                scheduler_name, base_scheduler_name, max_iter
            )
            logging.error(exp)
            raise exp
