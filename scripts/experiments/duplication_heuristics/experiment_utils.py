from saga.utils.random_graphs import (
    get_network, 
    get_branching_dag,
    get_chain_dag,
    get_diamond_dag,
    get_fork_dag
)
from saga import Network
from scripts.experiments.duplication_heuristics.heuristics import (
    is_super_node, task_score, compute_schedule_features, task_can_be_scored, processor_dup_benefits
)
from saga import TaskGraph, Schedule
import numpy as np
from copy import deepcopy

ALLOWED_DUPLICATES: set[str] = set()

# HELPER METHODS
def get_random_instance(ccr: float, levels: int, branching_factor: int, num_nodes: int, dag_type: str = "branching") -> tuple[Network, TaskGraph]:
    """
    Generate a random workflow and processor network with the requested CCR.
    """
    network = get_network(num_nodes=num_nodes)

    if dag_type == "branching":
        task_graph = get_branching_dag(levels=levels, branching_factor=branching_factor)

    elif dag_type == "fork":
        task_graph = get_fork_dag()

    elif dag_type == "diamond":
        task_graph = get_diamond_dag()

    else:
        raise ValueError(
            f"Unknown dag_type: {dag_type}"
        )

    network = network.scale_to_ccr(task_graph, ccr)
    return network, task_graph

def select_should_duplicate(task_name: str, task_graph: TaskGraph, network) -> bool:
    """Called by the scheduler to decide whether a task is allowed to duplicate."""
    return task_name in ALLOWED_DUPLICATES

def get_dup_plan(selected_tasks: list[str], task_graph: TaskGraph, schedule: Schedule) -> dict[str, list[str]]:
    """
    Determine the beneficial target processors for the selected tasks.
    
    Returns a duplication plan mapping each task to the processors where creating a duplicate 
    is estimated to reduce communication delays.

    Return example: {"task_name": ["P2", "P3"]}
    """
    plan = {}

    for task_name in selected_tasks:
        targets = processor_dup_benefits(task_name, task_graph, schedule)
        chosen_processors = [proc for proc, benefit in targets if benefit > 0]
        if chosen_processors: 
            plan[task_name] = chosen_processors
    return plan

def iterative_task_scores(
    baseline_schedule: Schedule,
    task_graph: TaskGraph,
    network: Network,
    scheduler_class,
    mode: str,
    max_top_n: int = 3
) -> list[dict]:
    """
    Iteratively select and duplicate the highest/lowest scoring tasks.
    
    After each task duplication, the schedule and heuristic scores are recomputed before selecting the next task.
    Returns the results from each duplication iteration.
    """

    global ALLOWED_DUPLICATES
    ALLOWED_DUPLICATES.clear()

    current_schedule = baseline_schedule
    duplicated_tasks: list[str] = []
    duplication_plan: dict[str, list[str]] = {}
    iteration_results: list[dict] = []

    for n in range(1, max_top_n + 1):
        schedule_features = compute_schedule_features(task_graph, current_schedule)
        scored_candidates = []

        for task in task_graph.tasks:
            task_name = task.name

            if (is_super_node(task_name)
                or not task_can_be_scored(task_name, task_graph)
                or task_name in duplicated_tasks
            ):
                continue

            raw = schedule_features["raw"][task_name]

            if (raw["num_target_processors"] <= 0 or raw["estimated_benefit"] <= 0):
                continue

            score = task_score(
                task_name,
                task_graph,
                network,
                current_schedule,
                schedule_features
            )

            scored_candidates.append({
                "task_name": task_name,
                "task_score": score,
                "estimated_benefit": raw["estimated_benefit"]
            })

        if not scored_candidates:
            break

        scored_candidates.sort(
            key=lambda candidate: (
                candidate["task_score"],
                candidate["estimated_benefit"]
            ),
            reverse=(mode == "top-n")
        )

        selected_candidate = None
        candidate_dup_plan = None

        # select the first ranked task with a valid duplication target
        for candidate in scored_candidates:
            candidate_task = candidate["task_name"]
            plan = get_dup_plan([candidate_task], task_graph, current_schedule)

            if candidate_task in plan:
                selected_candidate = candidate
                candidate_dup_plan = plan
                break

        if selected_candidate is None or candidate_dup_plan is None:
            break

        duplicated_task = selected_candidate["task_name"]
        duplicated_tasks.append(duplicated_task)

        # keep targets from previous iterations
        duplication_plan.update(candidate_dup_plan)
        ALLOWED_DUPLICATES.clear()
        ALLOWED_DUPLICATES.update(duplication_plan.keys())

        dup_factor = 1 + max(
            (len(targets) for targets in duplication_plan.values()),
            default=0
        )

        scheduler = scheduler_class(
            duplication_factor=dup_factor,
            duplication_targets={
                task_name: targets.copy() for task_name, targets
                in duplication_plan.items()
            }
        )

        current_schedule = scheduler.schedule(network, task_graph)

        iteration_results.append({
            "n": n,
            "task_name": duplicated_task,
            "task_score": selected_candidate["task_score"],
            "estimated_benefit": selected_candidate["estimated_benefit"],
            "duplicated_tasks": duplicated_tasks.copy(),
            "dup_factor": dup_factor,
            "schedule": current_schedule
        })

    return iteration_results

def compute_duplicated_task_stats(task_name: str, task_graph: TaskGraph, schedule: Schedule | None = None) -> dict:
    """Computes stats of tasks that got duplicated."""
    task = task_graph.get_task(task_name)

    incoming_edges = [
        edge for edge in task_graph.in_edges(task_name)
        if not is_super_node(edge.source)
    ]

    outgoing_edges = [
        edge for edge in task_graph.out_edges(task_name)
        if not is_super_node(edge.target)
    ]

    incoming = sum(edge.size for edge in incoming_edges)
    outgoing = sum(edge.size for edge in outgoing_edges)

    stats = {
        "task_name": task_name,
        "task_cost": task.cost,
        "incoming_comm": incoming,
        "outgoing_comm": outgoing,
        "in_out_ratio": ( float("inf") if outgoing == 0 else incoming / outgoing ),
        "out_in_ratio": ( float("inf") if incoming == 0 else outgoing / incoming ),
        "num_children": len(outgoing_edges),
        "num_parents": len(incoming_edges)
    }

    if schedule is None:
        return stats

    task_instances = schedule.get_scheduled_tasks(task_name)

    if not task_instances:
        stats.update({
            "num_copies_scheduled": 0,
            "original_task_node": None,
            "original_task_end": np.nan,
            "max_duplicate_end": np.nan,
            "latest_duplicate_node": None
        })
        return stats

    original_task = task_instances[0]
    duplicate_tasks = task_instances[1:]

    if duplicate_tasks:
        latest_duplicate = max( duplicate_tasks, key=lambda task: task.end )
        max_duplicate_end = latest_duplicate.end
        latest_duplicate_node = latest_duplicate.node
    else:
        max_duplicate_end = np.nan
        latest_duplicate_node = None

    stats.update({
        "num_copies_scheduled": len(task_instances),
        "original_task_node": original_task.node,
        "original_task_end": original_task.end,
        "max_duplicate_end": max_duplicate_end,
        "latest_duplicate_node": latest_duplicate_node
    })

    return stats

def compute_workflow_stats(task_graph: TaskGraph, network: Network) -> dict:
    """Computes workflow stats used for experiment analysis."""
    workflow = task_graph.graph
    network_graph = network.graph

    tasks = [ task for task in workflow.nodes if not is_super_node(task) ]

    edges = [
        (source, target)
        for source, target in workflow.edges
        if not is_super_node(source) and not is_super_node(target)
    ]

    avg_task_cost = np.mean([ workflow.nodes[task]["weight"] for task in tasks ])
    avg_edge_size = np.mean([ workflow.edges[source, target]["weight"] for source, target in edges ])
    avg_processor_speed = np.mean([ network_graph.nodes[node]["weight"] for node in network_graph.nodes ])

    network_speeds = [
        network_graph.edges[source, target]["weight"]
        for source, target in network_graph.edges
        if source != target
    ]

    avg_network_speed = np.mean(network_speeds)
    avg_computation_time = avg_task_cost / avg_processor_speed
    avg_communication_time = avg_edge_size / avg_network_speed

    return {
        "ccr": avg_communication_time / avg_computation_time,
        "avg_task_cost": avg_task_cost,
        "avg_edge_size": avg_edge_size
    }

def greedy_brute_force_validation(
    baseline_schedule: Schedule,
    task_graph: TaskGraph,
    network: Network,
    scheduler_class,
    max_top_n: int = 3,
    candidate_pool_size: int | None = None
) -> list[dict]:
    """
    Evaluate every eligible duplication candidate to identify the best task.
    
    At each iteration, every candidate is tested individually from the current schedule, 
    and the task producing the lowest makespan is selected (task scores break ties). 
    Returns the results for all evaluated candidates. 
    """

    global ALLOWED_DUPLICATES
    ALLOWED_DUPLICATES.clear()

    current_schedule = baseline_schedule
    duplicated_tasks: list[str] = []
    duplication_plan: dict[str, list[str]] = {}
    brute_force_results: list[dict] = []

    for n in range(1, max_top_n + 1):
        current_makespan = current_schedule.makespan

        # recompute heuristic features using the current schedule
        schedule_features = compute_schedule_features(task_graph, current_schedule)
        scored_candidates = []

        for task in task_graph.tasks:
            task_name = task.name

            if (is_super_node(task_name)
                or not task_can_be_scored(task_name, task_graph)
                or task_name in duplicated_tasks
            ):
                continue

            raw = schedule_features["raw"].get(task_name)
            if raw is None or raw["num_target_processors"] <= 0 or raw["estimated_benefit"] <= 0:
                continue

            score = task_score(
                task_name,
                task_graph,
                network,
                current_schedule,
                schedule_features
            )

            scored_candidates.append({
                "task_name": task_name,
                "task_score": score,
                "estimated_benefit": raw["estimated_benefit"]
            })

        if not scored_candidates:
            break

        # rank candidates according to the heuristic
        scored_candidates.sort(
            key=lambda candidate: (
                candidate["task_score"],
                candidate["estimated_benefit"]
            ),
            reverse=True
        )

        for heuristic_rank, candidate in enumerate(scored_candidates, start=1):
            candidate["heuristic_rank"] = heuristic_rank

        # none means test every eligible candidate
        tested_candidates = (
            scored_candidates[:candidate_pool_size]
            if candidate_pool_size is not None
            else scored_candidates
        )

        candidate_results = []
        # test every candidate individually from the same current schedule
        for candidate in tested_candidates:
            candidate_task = candidate["task_name"]

            candidate_task_plan = get_dup_plan( [candidate_task], task_graph, current_schedule )
            if candidate_task not in candidate_task_plan:
                continue

            # keep previous winning duplications and temporarily add the candidate being tested
            candidate_duplication_plan = {
                task_name: targets.copy() for task_name, targets
                in duplication_plan.items()
            }

            candidate_duplication_plan.update({
                task_name: targets.copy() for task_name, targets
                in candidate_task_plan.items()
            })

            dup_factor = 1 + max(
                (len(targets) for targets in candidate_duplication_plan.values()),
                default=0
            )

            ALLOWED_DUPLICATES.clear()
            ALLOWED_DUPLICATES.update(candidate_duplication_plan.keys())

            scheduler = scheduler_class(
                duplication_factor=dup_factor,
                duplication_targets={
                    task_name: targets.copy() for task_name, targets
                    in candidate_duplication_plan.items()
                }
            )

            candidate_schedule = scheduler.schedule(network, task_graph)
            candidate_makespan = candidate_schedule.makespan

            candidate_results.append({
                "n": n,
                "task_name": candidate_task,
                "task_score": candidate["task_score"],
                "heuristic_rank": candidate["heuristic_rank"],
                "estimated_benefit": candidate["estimated_benefit"],
                "num_target_processors": len(candidate_task_plan[candidate_task]),
                "candidate_schedule": candidate_schedule,
                "candidate_duplication_plan": candidate_duplication_plan,
                "current_makespan": current_makespan,
                "candidate_makespan": candidate_makespan,
                "makespan_ratio_current": (candidate_makespan / current_makespan),
                "makespan_ratio_baseline": (candidate_makespan / baseline_schedule.makespan)
            })

        if not candidate_results:
            break

        # lowest resulting makespan is best. task score breaks ties
        candidate_results.sort(
            key=lambda candidate: ( candidate["candidate_makespan"], -candidate["task_score"] )
        )

        for brute_force_rank, candidate in enumerate(candidate_results, start=1):
            candidate["brute_force_rank"] = brute_force_rank
            candidate["selected_by_brute_force"] = (brute_force_rank == 1)

        best_result = candidate_results[0]
        best_task = best_result["task_name"]
        duplicated_tasks.append(best_task)

        # keep the winning duplication plan for the next iteration
        duplication_plan = {
            task_name: targets.copy() for task_name, targets
            in best_result["candidate_duplication_plan"].items()
        }

        current_schedule = best_result["candidate_schedule"]
        heuristic_top_task = scored_candidates[0]["task_name"]
        heuristic_matches_brute_force = (heuristic_top_task == best_task)

        # save every tested candidate
        for candidate in candidate_results:
            brute_force_results.append({
                "n": candidate["n"],
                "task_name": candidate["task_name"],
                "task_score": candidate["task_score"],
                "heuristic_rank": candidate["heuristic_rank"],
                "estimated_benefit": candidate["estimated_benefit"],
                "num_target_processors": candidate["num_target_processors"],
                "brute_force_rank": candidate["brute_force_rank"],
                "selected_by_brute_force": candidate["selected_by_brute_force"],
                "heuristic_top_task": heuristic_top_task,
                "best_task_this_iteration": best_task,
                "heuristic_top_matches_brute_force": heuristic_matches_brute_force,
                "duplicated_tasks": duplicated_tasks.copy(),
                "current_makespan": candidate["current_makespan"],
                "candidate_makespan": candidate["candidate_makespan"],
                "makespan_ratio_current": candidate["makespan_ratio_current"],
                "makespan_ratio_baseline": candidate["makespan_ratio_baseline"],
                # needed to compute schedule diagnostics outside this function
                "candidate_schedule": candidate["candidate_schedule"]
            })

    ALLOWED_DUPLICATES.clear()
    return brute_force_results
