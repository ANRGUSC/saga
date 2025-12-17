"""PISA experiments: finding adversarial scheduling instances using simulated annealing."""
import logging
import pathlib
import random
from itertools import product
from typing import List, Optional, Set, Tuple

from saga.pisa import (
    SCHEDULERS,
    SchedulerName,
    SimulatedAnnealing,
    SimulatedAnnealingConfig,
)
from saga.pisa.changes import (
    NetworkChangeEdgeWeight,
    NetworkChangeNodeWeight,
    TaskGraphAddDependency,
    TaskGraphChangeDependencyWeight,
    TaskGraphChangeTaskWeight,
    TaskGraphDeleteDependency,
)
from saga.utils.random_graphs import get_chain_dag, get_network
from saga.utils.random_variable import RandomVariable, UniformRandomVariable

thisdir = pathlib.Path(__file__).parent

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)

# Algorithms that only work when the compute speed is the same for all nodes
HOMOGENOUS_COMP_ALGS: Set[str] = {"ETF", "FCP", "FLB"}
# Algorithms that only work when the communication speed is the same for all network edges
HOMOGENOUS_COMM_ALGS: Set[str] = {"BIL", "GDL", "FCP", "FLB"}


def run_experiments(
    scheduler_pairs: List[Tuple[SchedulerName, SchedulerName]],
    max_iterations: int = 1000,
    num_tries: int = 10,
    max_temp: float = 10.0,
    min_temp: float = 0.1,
    cooling_rate: float = 0.99,
    skip_existing: bool = True,
    output_path: Optional[pathlib.Path] = None,
    node_range: Tuple[int, int] = (3, 5),
    task_range: Tuple[int, int] = (3, 5),
) -> None:
    """Run PISA experiments for finding adversarial scheduling instances.

    Args:
        scheduler_pairs: List of (scheduler, base_scheduler) name pairs to test.
        max_iterations: Maximum iterations per SA run.
        num_tries: Number of random restarts per scheduler pair.
        max_temp: Starting temperature.
        min_temp: Stopping temperature.
        cooling_rate: Temperature decay rate per iteration.
        skip_existing: Skip experiments that already have results.
        output_path: Directory for results (default: {thisdir}/results).
        node_range: Range for random network node count.
        task_range: Range for random task graph task count.
    """
    output_path = output_path or thisdir / "results"
    output_path.mkdir(parents=True, exist_ok=True)

    for scheduler_name, base_scheduler_name in scheduler_pairs:
        # Skip self-comparisons
        if scheduler_name == base_scheduler_name:
            continue

        run_name = f"{base_scheduler_name}_vs_{scheduler_name}"
        run_dir = output_path / run_name

        # Check if we should skip
        if skip_existing and run_dir.exists():
            run_json = run_dir / "run.json"
            if run_json.exists():
                logging.info("Skipping existing: %s", run_name)
                continue

        logging.info("Running: %s", run_name)

        best_sa: Optional[SimulatedAnnealing] = None
        best_energy: float = 0.0

        for try_num in range(num_tries):
            try:
                change_type_names: List[str] = [
                    TaskGraphAddDependency.__name__,
                    TaskGraphDeleteDependency.__name__,
                    TaskGraphChangeDependencyWeight.__name__,
                    TaskGraphChangeTaskWeight.__name__,
                ]

                # Generate random initial problem instance
                num_nodes = random.randint(*node_range)
                num_tasks = random.randint(*task_range)
                node_weight_distribution = UniformRandomVariable(0.1, 1.0)
                edge_weight_distribution = UniformRandomVariable(0.1, 1.0)
                if scheduler_name in HOMOGENOUS_COMP_ALGS or base_scheduler_name in HOMOGENOUS_COMP_ALGS:
                    node_weight_distribution = RandomVariable(samples=[1.0])
                else:
                    change_type_names.append(NetworkChangeEdgeWeight.__name__)
                if scheduler_name in HOMOGENOUS_COMM_ALGS or base_scheduler_name in HOMOGENOUS_COMM_ALGS:
                    edge_weight_distribution = RandomVariable(samples=[1.0])
                else:
                    change_type_names.append(NetworkChangeNodeWeight.__name__)
                network = get_network(
                    num_nodes=num_nodes,
                    node_weight_distribution=node_weight_distribution,
                    edge_weight_distribution=edge_weight_distribution,
                )
                task_graph = get_chain_dag(num_tasks)

                # Create config
                config = SimulatedAnnealingConfig(
                    max_iterations=max_iterations,
                    max_temp=max_temp,
                    min_temp=min_temp,
                    cooling_rate=cooling_rate,
                    change_types=change_type_names,
                )

                # Create unique name for this try
                try_name = f"{run_name}_try{try_num}"

                # Run simulated annealing
                sa = SimulatedAnnealing(
                    name=try_name,
                    scheduler=scheduler_name,
                    base_scheduler=base_scheduler_name,
                    initial_network=network,
                    initial_task_graph=task_graph,
                    config=config,
                    data_dir=output_path / ".runs",
                )

                result = sa.execute(progress=True)

                # Track best result
                if result.best_energy > best_energy:
                    best_energy = result.best_energy
                    best_sa = sa
                    logging.info(
                        "  Try %d: New best energy %.4f",
                        try_num, best_energy
                    )
                else:
                    logging.info(
                        "  Try %d: Energy %.4f (best: %.4f)",
                        try_num, result.best_energy, best_energy
                    )

            except Exception as e:
                logging.error(
                    "Error in try %d for %s: %s",
                    try_num, run_name, e
                )
                raise

        # Save best result to main results directory
        if best_sa is not None:
            # Copy best run to main results location
            import shutil
            best_run_dir = output_path / ".runs" / best_sa.name
            final_dir = output_path / run_name
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.copytree(best_run_dir, final_dir)
            logging.info("Saved best result for %s (energy: %.4f)", run_name, best_energy)


def main():
    """Run PISA experiments."""
    results_dir = thisdir / "results"

    # Get all scheduler names from the registry
    scheduler_names: List[SchedulerName] = list(SCHEDULERS.keys())  # type: ignore

    run_experiments(
        scheduler_pairs=[
            (s1, s2) for s1, s2 in product(scheduler_names, scheduler_names)
            if s1 != s2
        ],
        max_iterations=1000,
        num_tries=10,
        max_temp=10.0,
        min_temp=0.1,
        cooling_rate=0.99,
        skip_existing=True,
        output_path=results_dir,
    )


if __name__ == "__main__":
    main()
