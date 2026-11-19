import itertools
import pathlib
from .simulated_annealing import DupSimulatedAnnealingConfig, DupSimulatedAnnealing
from saga.schedulers.data.wfcommons import get_wfcommons_instance
from ..heuristics import is_super_node
from saga.schedulers import HeftScheduler, CpopScheduler
from saga import TaskGraph, Network
import hashlib
import random
import numpy as np
import json
from tqdm.auto import tqdm

SCHEDULERS = {
    "HEFT": HeftScheduler,
    "CPoP": CpopScheduler
}

def compute_dag_averages(task_graph: TaskGraph) -> dict:
    """Compute average structural and cost features of a DAG."""

    tasks = [
        task for task in task_graph.tasks
        if not is_super_node(task.name)
    ]

    task_names = {task.name for task in tasks}

    edges = [
        edge for edge in task_graph.dependencies
        if edge.source in task_names and edge.target in task_names
    ]

    num_tasks = len(tasks)
    num_edges = len(edges)

    return {
        "avg_task_cost": (
            sum(task.cost for task in tasks) / num_tasks
            if num_tasks else 0.0
        ),
        "avg_edge_cost": (
            sum(edge.size for edge in edges) / num_edges
            if num_edges else 0.0
        ),
        "avg_num_parents": (
            num_edges / num_tasks if num_tasks else 0.0
        ),
        "avg_num_children": (
            num_edges / num_tasks if num_tasks else 0.0
        )
    }

def get_task_species(task_name: str) -> str:
    """Extract the task type from a WfCommons task name."""
    return task_name.rsplit("_", 1)[0]

def build_species_analysis(task_graph: TaskGraph, best_dup_plan: dict[str, list[str]]) -> dict:
    """Compare duplicated tasks with other tasks of the same species."""

    duplicated_species = {
        get_task_species(task_name)
        for task_name in best_dup_plan
    }

    species_analysis = {}

    for task in task_graph.tasks:
        if is_super_node(task.name):
            continue

        species = get_task_species(task.name)

        if species not in duplicated_species:
            continue

        incoming_edges = [
            edge for edge in task_graph.dependencies
            if edge.target == task.name
            and not is_super_node(edge.source)
        ]

        outgoing_edges = [
            edge for edge in task_graph.dependencies
            if edge.source == task.name
            and not is_super_node(edge.target)
        ]

        task_data = {
            "task_name": task.name,
            "duplicated": task.name in best_dup_plan,
            "target_processors": best_dup_plan.get(task.name, []),
            "num_parents": len(incoming_edges),
            "num_children": len(outgoing_edges),
            "task_cost": task.cost,
            "incoming_data_total": sum(
                edge.size for edge in incoming_edges
            ),
            "outgoing_data_total": sum(
                edge.size for edge in outgoing_edges
            )
        }

        species_analysis.setdefault(species, []).append(task_data)

    return {"species": species_analysis}

def build_summary(results_dir: pathlib.Path) -> None:
    """Build a JSON summary of completed SA experiments."""
    runs_dir = results_dir / "runs"
    summaries = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # Only include completed experiments
        if not (run_dir / "completed.json").exists():
            continue

        sa = DupSimulatedAnnealing.load(
            name=run_dir.name,
            data_dir=runs_dir
        )

        recipe, scheduler, ccr, processors, trial = sa.name.rsplit("_", 4)

        ccr = float(ccr.removeprefix("ccr"))
        processors = int(processors.removeprefix("p"))
        trial = int(trial.removeprefix("t"))

        last_iteration = sa.get_last_iteration()
        baseline_makespan = sa.get_state({}).makespan
        dag_avgs = compute_dag_averages(sa.task_graph)

        if last_iteration is None:
            # No eligible duplication moves
            iterations_completed = 0
            best_makespan = baseline_makespan
            best_dup_plan = {}
            best_iteration = None
        else:
            iterations_completed = last_iteration.iteration + 1
            best_makespan = last_iteration.best_makespan
            best_dup_plan = last_iteration.best_dup_plan
            best_iteration = last_iteration.best_iteration

        species_analysis = build_species_analysis(sa.task_graph, best_dup_plan)
        species_path = run_dir / "species_analysis.json"
        species_path.write_text(json.dumps(species_analysis, indent=2))
        summaries.append({
            "name": sa.name,
            "recipe": recipe,
            "scheduler": scheduler,
            "ccr": ccr,
            "num_processors": processors,
            "trial": trial,
            "instance_seed": sa.instance_seed,
            "sa_seed": sa.sa_seed,
            "num_tasks": len(sa.task_graph.tasks),
            "iterations_completed": iterations_completed,
            "baseline_makespan": baseline_makespan,
            "best_makespan": best_makespan,
            "best_makespan_ratio": best_makespan / baseline_makespan,
            "best_iteration": best_iteration,
            "best_dup_plan": best_dup_plan,
            "avg_task_cost": dag_avgs["avg_task_cost"],
            "avg_edge_cost": dag_avgs["avg_edge_cost"],
            "avg_num_parents": dag_avgs["avg_num_parents"],
            "avg_num_children": dag_avgs["avg_num_children"]
        })

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))

def make_seed(*values) -> int:
    """Generate a stable seed from experiment parameters."""
    key = "|".join(map(str, values))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")

def get_shared_instance(recipe: str, ccr: float, processors: int, trial: int, instances_dir: pathlib.Path) -> tuple[Network, TaskGraph]:
    """"""
    instance_name = f"{recipe}_ccr{ccr:g}_p{processors}_t{trial:03d}" 
    instance_dir = instances_dir / instance_name
    network_path = instance_dir / "network.json"
    task_graph_path = instance_dir / "task_graph.json"

    if network_path.exists() and task_graph_path.exists():
        return (
            Network.model_validate_json(network_path.read_text()),
            TaskGraph.model_validate_json(task_graph_path.read_text())
        )

    if network_path.exists() or task_graph_path.exists():
        raise RuntimeError(f"Incomplete shared instance: {instance_dir}")

    instance_seed = make_seed(recipe, ccr, processors, trial, "instance")
    random.seed(instance_seed)
    np.random.seed(instance_seed)

    network, task_graph = get_wfcommons_instance(
        recipe_name=recipe,
        ccr=ccr,
        max_size_multiplier=5,
        num_nodes=processors
    )

    instance_dir.mkdir(parents=True, exist_ok=True)
    network_path.write_text(network.model_dump_json(indent=2))
    task_graph_path.write_text(task_graph.model_dump_json(indent=2))

    return network, task_graph

def run_multi_experiments():
    """"""
    # add Prof Coleman to fork repo + work on paper
    recipes = [
        "cycles", "genome", "blast", 
        "bwa", "srasearch", "seismology", "soykb"
    ]
    ccrs = [0.1, 1.0, 5.0]
    num_processors = [3, 5, 8]
    schedulers = ["HEFT", "CPoP"]
    num_trials = 10

    results_dir = pathlib.Path(__file__).parent / "heuristic_results_new"
    runs_dir = results_dir / "runs"
    instances_dir = pathlib.Path(__file__).parent.parent / "graph_instances"

    combinations = itertools.product(
        recipes,
        schedulers,
        ccrs,
        num_processors,
        range(num_trials)
    )

    total_experiments = (
        len(recipes) * 
        len(schedulers) * 
        len(ccrs) * 
        len(num_processors) * 
        num_trials
    )

    for recipe, scheduler_name, ccr, processors, trial in tqdm(
        combinations, total=total_experiments, desc="Overall experiments"
    ):
        run_name = (
            f"{recipe}_{scheduler_name}_"
            f"ccr{ccr:g}_p{processors}_t{trial:03d}"
        )
        print(f"Run name: {run_name}")
        instance_seed = make_seed(recipe, ccr, processors, trial, "instance")
        sa_seed = make_seed(recipe, ccr, processors, trial, "sa")
        run_path = runs_dir / run_name

        if (run_path / "run.json").exists():
            sa = DupSimulatedAnnealing.load(name=run_name, data_dir=runs_dir)
        else:
            network, task_graph = get_shared_instance(
                recipe=recipe,
                ccr=ccr,
                processors=processors,
                trial=trial,
                instances_dir=instances_dir
            )

            config = DupSimulatedAnnealingConfig(
                max_iterations=1000,  
                max_temp=100.0,
                min_temp=0.1,
                cooling_rate=0.99
            )

            sa = DupSimulatedAnnealing(
                name=run_name,
                data_dir=runs_dir,
                network=network,
                task_graph=task_graph,
                scheduler_class=SCHEDULERS[scheduler_name],
                config=config,
                instance_seed=instance_seed,
                sa_seed=sa_seed
            )

            sa.save()
            random.seed(sa_seed)

        sa.execute(progress=False)


if __name__ == "__main__":
    run_multi_experiments()
    results_dir = pathlib.Path(__file__).parent / "heuristic_results_new"
    build_summary(results_dir)