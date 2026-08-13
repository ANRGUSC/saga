"""Evaluate whether task_score predicts good duplication targets.

For each random instance, rank tasks by task_score, then measure makespan when
duplicating the top-N vs the bottom-N tasks (N = 0..max). If the heuristic is
useful, duplicating top-N tasks should reduce makespan more (or hurt less) than
duplicating bottom-N tasks.

The duplication decision is injected by monkeypatching should_duplicate in the
HEFT and CPOP modules to allow only a chosen set of tasks; the schedulers are not
modified.
"""

from saga.schedulers import HeftScheduler, CpopScheduler
import saga.schedulers.heft as heft_mod
import saga.schedulers.cpop as cpop_mod
from scripts.experiments.duplication_heuristics.heuristics import is_super_node
from scripts.experiments.duplication_heuristics.experiment_utils import (
    get_random_instance, select_should_duplicate, iterative_task_scores, 
    compute_duplicated_task_stats, compute_workflow_stats, greedy_brute_force_validation
)
from itertools import product
from tqdm import tqdm
import pandas as pd
import pathlib
from saga.schedulers.data.wfcommons import get_wfcommons_instance

thisdir = pathlib.Path(__file__).parent.resolve()
savedir = thisdir / "outputs" 

MAX_TOP_N = 3
RUN_BRUTE_FORCE = True # switch to false if brute-force experiments not needed

def random_graph_experiments():
    """Run heuristic and brute-force experiments on small random graphs."""
    heft_mod.should_duplicate = select_should_duplicate
    cpop_mod.should_duplicate = select_should_duplicate

    # experiment inputs
    levels = [3, 4, 5]
    branching_factors = [2, 3, 4]
    num_processors = [2, 4, 8]
    num_experiments = 40
    ccr_values = [0.1, 1, 10]
    modes = ["top-n", "bottom-n"]
    scheduler_classes = [("heft", HeftScheduler), ("cpop", CpopScheduler)]
    # fork and diamond will be added once their functions are fixed
    dags = ["branching"]

    total_iterations = (
        num_experiments
        * len(levels)
        * len(branching_factors)
        * len(ccr_values)
        * len(num_processors)
        * len(scheduler_classes)
        * (len(modes) * MAX_TOP_N + (1 if RUN_BRUTE_FORCE else 0))
    )

    rand_graphs_data = []
    brute_force_data = []

    with tqdm(total=total_iterations, desc="Running random graphs experiments") as pbar:
        for dag_name in dags:
            for experiment_num in range(num_experiments):
                for ccr in ccr_values:
                    for num_processor in num_processors:
                        level_branch_combinations = (
                            product(levels, branching_factors)
                            if dag_name == "branching" else [(0, 0)]
                        )

                        for level, branch_factor in level_branch_combinations:
                            network, task_graph = get_random_instance(
                                ccr,
                                level,
                                branch_factor,
                                num_processor,
                                dag_name
                            )

                            num_tasks = len([
                                task for task in task_graph.tasks
                                if not is_super_node(task.name)
                            ])

                            for scheduler_name, scheduler_class in scheduler_classes:
                                baseline_scheduler = scheduler_class(duplication_factor=1)
                                baseline_schedule = baseline_scheduler.schedule(network, task_graph)
                                baseline_makespan = baseline_schedule.makespan

                                # brute-force done to validate heuristics
                                if RUN_BRUTE_FORCE:
                                    brute_force_results = greedy_brute_force_validation(
                                        baseline_schedule=baseline_schedule,
                                        task_graph=task_graph,
                                        network=network,
                                        scheduler_class=scheduler_class,
                                        max_top_n=MAX_TOP_N,
                                        candidate_pool_size=None
                                    )

                                    for result in brute_force_results:
                                        task_name = result["task_name"]
                                        task_stats = compute_duplicated_task_stats(task_name, task_graph)

                                        brute_force_data.append({
                                            "scheduler_name": scheduler_name,
                                            "experiment_num": experiment_num,
                                            "dag_type": dag_name,
                                            "num_tasks": num_tasks,
                                            "ccr": ccr,
                                            "levels": level,
                                            "branching_factor": branch_factor,
                                            "num_processors": num_processor,
                                            "n": result["n"],
                                            "task_name": task_name,
                                            "task_score": result["task_score"],
                                            "heuristic_rank": result["heuristic_rank"], # rank assigned by the heuristic
                                            "heuristic_top_task": result["heuristic_top_task"], # highest-scoring heuristic rank
                                            "brute_force_rank": result["brute_force_rank"], # rank based on resulting makespan
                                            "selected_by_brute_force": result["selected_by_brute_force"], # true if this task produced the best makespan
                                            "best_task_this_iteration": result["best_task_this_iteration"], # task selected by brute-force
                                            "heuristic_top_matches_brute_force": result["heuristic_top_matches_brute_force"], # true if heuristics top task matches brute-force winner
                                            "duplicated_tasks": ",".join(result["duplicated_tasks"]), 
                                            "estimated_benefit": result["estimated_benefit"], # estimated benefit of duplicating this task
                                            "baseline_makespan": baseline_makespan, 
                                            "current_makespan": result["current_makespan"], # makespan before duplicating this task
                                            "candidate_makespan": result["candidate_makespan"], # makespan after duplicating this task
                                            "makespan_ratio_current": result["makespan_ratio_current"], # candidate_makespan / current_makespan
                                            "makespan_ratio_baseline": result["makespan_ratio_baseline"], # candidate_makespan / baseline_makespan
                                            **task_stats
                                        })
                                    pbar.update(1)

                                # regular heuristic experiments
                                for mode in modes:
                                    scoring_results = iterative_task_scores(
                                        baseline_schedule=baseline_schedule,
                                        task_graph=task_graph,
                                        network=network,
                                        scheduler_class=scheduler_class,
                                        mode=mode,
                                        max_top_n=MAX_TOP_N
                                    )

                                    for result in scoring_results:
                                        task_name = result["task_name"]
                                        schedule = result["schedule"]
                                        task_stats = compute_duplicated_task_stats(task_name, task_graph)

                                        rand_graphs_data.append({
                                            "scheduler_name": scheduler_name,
                                            "experiment_num": experiment_num,
                                            "num_tasks": num_tasks,
                                            "ccr": ccr,
                                            "levels": level,
                                            "branching_factor": branch_factor,
                                            "num_processors": num_processor,
                                            "mode": mode,
                                            "n": result["n"],
                                            "task_name": task_name,
                                            "task_score": result["task_score"],
                                            "duplicated_tasks": ",".join( result["duplicated_tasks"]),
                                            "estimated_benefit": result["estimated_benefit"],
                                            "dup_factor": result["dup_factor"],
                                            "makespan": schedule.makespan,
                                            "baseline_makespan": baseline_makespan,
                                            "makespan_ratio": (schedule.makespan / baseline_makespan),
                                            **task_stats
                                        })

                                        pbar.update(1)

    rand_graphs_df = pd.DataFrame(rand_graphs_data)
    rand_graphs_df.to_csv(savedir / "rand_graphs_data.csv", index=False)
    brute_force_df = pd.DataFrame(brute_force_data)
    brute_force_df.to_csv(savedir / "rand_graphs_brute_force_data.csv", index=False)

def wfcommons_experiments():
    """Run heuristic and brute-force experiments on WfCommons workflows."""
    heft_mod.should_duplicate = select_should_duplicate
    cpop_mod.should_duplicate = select_should_duplicate

    num_workflows = 40
    modes = ["top-n", "bottom-n"]
    ccr_vals = [0.1, 0.5, 1, 5, 10]
    num_processors = [4, 8, 16]
    workflow_recipes = ["seismology", "montage", "epigenomics"]
    schedulers = [("heft", HeftScheduler), ("cpop", CpopScheduler)]
    total_iterations = (
        len(workflow_recipes)
        * num_workflows
        * len(ccr_vals)
        * len(num_processors)
        * len(schedulers)
        * (len(modes) * MAX_TOP_N + (1 if RUN_BRUTE_FORCE else 0))
    )

    wfcommons_data = []
    wfcommons_brute_force_data = []

    with tqdm(total=total_iterations, desc="Running WfCommons experiments") as pbar:
        for recipe in workflow_recipes:
            for workflow_instance in range(num_workflows):
                for ccr, num_processor in product(ccr_vals, num_processors):
                    network, task_graph = get_wfcommons_instance(
                        recipe_name=recipe,
                        ccr=ccr,
                        max_size_multiplier=14,
                        num_nodes=num_processor
                    )

                    workflow_stats = compute_workflow_stats(task_graph, network)
                    avg_task_cost = workflow_stats["avg_task_cost"]
                    avg_edge_size = workflow_stats["avg_edge_size"]
                    num_tasks = len([
                        task for task in task_graph.tasks
                        if not is_super_node(task.name)
                    ])

                    for scheduler_name, scheduler_class in schedulers:
                        baseline_scheduler = scheduler_class(duplication_factor=1)
                        baseline_schedule = baseline_scheduler.schedule(network, task_graph)
                        baseline_makespan = baseline_schedule.makespan

                        # brute-force testing only on a pool of 10 tasks so that runtime is not too slow
                        if RUN_BRUTE_FORCE:
                            brute_force_results = greedy_brute_force_validation(
                                baseline_schedule=baseline_schedule,
                                task_graph=task_graph,
                                network=network,
                                scheduler_class=scheduler_class,
                                max_top_n=MAX_TOP_N,
                                candidate_pool_size=10
                            )

                            for result in brute_force_results:
                                task_name = result["task_name"]
                                task_stats = compute_duplicated_task_stats(task_name, task_graph, result["candidate_schedule"])
                                wfcommons_brute_force_data.append({
                                    "scheduler_name": scheduler_name,
                                    "recipe": recipe,
                                    "workflow_instance": workflow_instance,
                                    "ccr": ccr,
                                    "num_processors": num_processor,
                                    "n": result["n"],
                                    "task_name": task_name,
                                    "task_score": result["task_score"],
                                    "heuristic_rank": result["heuristic_rank"], # rank assigned by the heuristic
                                    "heuristic_top_task": result["heuristic_top_task"], # highest-scoring heuristic rank
                                    "brute_force_rank": result["brute_force_rank"], # rank based on resulting makespan
                                    "selected_by_brute_force": result["selected_by_brute_force"], # true if this task produced the best makespan
                                    "best_task_this_iteration": result["best_task_this_iteration"], # task selected by brute-force
                                    "heuristic_top_matches_brute_force": result["heuristic_top_matches_brute_force"], # true if heuristics top task matches brute-force winner
                                    "duplicated_tasks": ",".join(result["duplicated_tasks"]), 
                                    "estimated_benefit": result["estimated_benefit"], # estimated benefit of duplicating this task
                                    "num_target_processors": result["num_target_processors"],
                                    "baseline_makespan": baseline_makespan,
                                    "current_makespan": result["current_makespan"], # makespan before duplicating this task
                                    "candidate_makespan": result["candidate_makespan"], # makespan after duplicating this task
                                    "makespan_ratio_current": result["makespan_ratio_current"], # candidate_makespan / current_makespan
                                    "makespan_ratio_baseline": result["makespan_ratio_baseline"], # candidate_makespan / baseline_makespan
                                    "num_tasks": num_tasks,
                                    "avg_task_cost": avg_task_cost,
                                    "avg_edge_size": avg_edge_size,
                                    **task_stats
                                })
                            pbar.update(1)

                        # regular heuristic experiments
                        for mode in modes:
                            scoring_results = iterative_task_scores(
                                baseline_schedule,
                                task_graph,
                                network,
                                scheduler_class,
                                mode,
                                MAX_TOP_N
                            )

                            for result in scoring_results:
                                n = result["n"]
                                duplicated_tasks = result["duplicated_tasks"]
                                duplicated_record = result["duplicated_record"]
                                dup_factor = result["dup_factor"]
                                schedule = result["schedule"]
                                duplicated_task = duplicated_tasks[-1]
                                task_stats = compute_duplicated_task_stats(duplicated_task, task_graph)
                                wfcommons_data.append({
                                    "scheduler_name": scheduler_name,
                                    "recipe": recipe,
                                    "workflow_instance": workflow_instance,
                                    "ccr": ccr,
                                    "num_processors": num_processor,
                                    "mode": mode,
                                    "n": n,
                                    "task_name": duplicated_task,
                                    "task_score": duplicated_record["score"],
                                    "duplicated_tasks": ",".join(duplicated_tasks),
                                    "estimated_benefit": duplicated_record["estimated_benefit"],
                                    "dup_factor": dup_factor,
                                    "makespan": schedule.makespan,
                                    "baseline_makespan": baseline_makespan,
                                    "makespan_ratio": (schedule.makespan / baseline_makespan),
                                    "num_tasks": num_tasks,
                                    "avg_task_cost": avg_task_cost,
                                    "avg_edge_size": avg_edge_size,
                                    **task_stats
                                })

                                pbar.update(1)

    wfcommons_df = pd.DataFrame(wfcommons_data)
    wfcommons_brute_force_df = pd.DataFrame(wfcommons_brute_force_data)
    wfcommons_df.to_csv(savedir / "wfcommons_data.csv", index=False)
    wfcommons_brute_force_df.to_csv(savedir / "wfcommons_brute_force_data.csv", index=False)

def main():
    #random_graph_experiments()
    wfcommons_experiments()

if __name__ == "__main__":
    main()