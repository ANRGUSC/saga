"""PISA experiments: finding adversarial scheduling instances using simulated annealing."""
import argparse
import json
import logging
import pathlib
import random
import shutil
from itertools import product
from typing import List, Optional, Sequence, Set, Tuple

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
    keep_runs: bool = False,
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
        keep_runs: Keep the per-try working directories under `.runs` after the
            best try of each pair has been copied into the results directory.
            Every iteration of every try is serialized to disk, so keeping them
            costs roughly `num_tries` times as much space as the results
            themselves (about 15GB for a full run with the default settings).
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

                sa.execute(progress=True)

                # Track best result
                if sa.best_iteration.current_energy > best_energy:
                    best_energy = sa.best_iteration.current_energy
                    best_sa = sa
                    logging.info(
                        "  Try %d: New best energy %.4f",
                        try_num, best_energy
                    )
                else:
                    logging.info(
                        "  Try %d: Energy %.4f (best: %.4f)",
                        try_num, sa.best_iteration.current_energy, best_energy
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
            best_run_dir = output_path / ".runs" / best_sa.name
            final_dir = output_path / run_name
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.copytree(best_run_dir, final_dir)

            # A run locates its iterations at {data_dir}/{name}, so the copy would
            # still point back at the working directory under .runs. Rewrite those
            # two fields so the saved result stands on its own.
            run_data = json.loads((final_dir / "run.json").read_text())
            run_data["name"] = run_name
            run_data["data_dir"] = str(output_path)
            (final_dir / "run.json").write_text(json.dumps(run_data, indent=2))

            logging.info("Saved best result for %s (energy: %.4f)", run_name, best_energy)

        # Discard this pair's working directories now that the best try is saved
        if not keep_runs:
            for try_num in range(num_tries):
                try_dir = output_path / ".runs" / f"{run_name}_try{try_num}"
                if try_dir.exists():
                    shutil.rmtree(try_dir)


def get_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run PISA experiments. The default reproduces the paper: every "
            "ordered pair of schedulers, 10 tries each. That takes about an "
            "hour on a fast machine and several hours on a small one, so use "
            "--quick for a demo."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  uv run python run.py --quick\n"
            "  uv run python run.py --schedulers HEFT CPoP OLB --num-tries 3\n"
            "  uv run python run.py\n"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Demo-sized run: 4 schedulers, 2 tries, 300 iterations. Finishes "
            "in about a minute. Overridden by the options below if given."
        ),
    )
    parser.add_argument(
        "--schedulers",
        nargs="+",
        metavar="NAME",
        choices=sorted(SCHEDULERS.keys()),
        help=(
            "Restrict the experiment to pairs drawn from these schedulers. "
            "Choices: " + ", ".join(sorted(SCHEDULERS.keys()))
        ),
    )
    parser.add_argument(
        "--num-tries",
        type=int,
        metavar="N",
        help="Random restarts per scheduler pair (default: 10).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        metavar="N",
        help="Maximum simulated annealing iterations per try (default: 1000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run pairs that already have results instead of skipping them.",
    )
    parser.add_argument(
        "--keep-runs",
        action="store_true",
        help=(
            "Keep every try's working directory under results/.runs. This is "
            "roughly num-tries times the size of the results (~15GB for a full "
            "run), so it is discarded by default."
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=thisdir / "results",
        metavar="DIR",
        help="Directory for results (default: ./results).",
    )
    return parser


QUICK_SCHEDULERS: Sequence[str] = ("HEFT", "CPoP", "OLB", "MinMin")


def main():
    """Run PISA experiments."""
    args = get_parser().parse_args()

    scheduler_names: List[SchedulerName] = list(SCHEDULERS.keys())  # type: ignore
    num_tries = 10
    max_iterations = 1000

    if args.quick:
        scheduler_names = list(QUICK_SCHEDULERS)  # type: ignore
        num_tries = 2
        max_iterations = 300

    # Explicit options win over the --quick preset
    if args.schedulers:
        scheduler_names = list(args.schedulers)
    if args.num_tries is not None:
        num_tries = args.num_tries
    if args.max_iterations is not None:
        max_iterations = args.max_iterations

    scheduler_pairs: List[Tuple[SchedulerName, SchedulerName]] = [
        (s1, s2) for s1, s2 in product(scheduler_names, scheduler_names) if s1 != s2
    ]
    logging.info(
        "Running %d scheduler pairs, %d tries each, up to %d iterations per try.",
        len(scheduler_pairs), num_tries, max_iterations,
    )

    run_experiments(
        scheduler_pairs=scheduler_pairs,
        max_iterations=max_iterations,
        num_tries=num_tries,
        max_temp=10.0,
        min_temp=0.1,
        cooling_rate=0.99,
        skip_existing=not args.overwrite,
        output_path=args.output,
        keep_runs=args.keep_runs,
    )


if __name__ == "__main__":
    main()
