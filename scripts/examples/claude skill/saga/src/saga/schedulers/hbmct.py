import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph
from saga.schedulers.heft import heft_rank_sort


def hbmct_rank_sort(network: Network, task_graph: TaskGraph) -> List[str]:
    """Sort tasks based on their rank (as defined in the HEFT paper).

    Args:
        network (Network): The network.
        task_graph (TaskGraph): The task graph.

    Returns:
        List[str]: The sorted list of tasks.
    """
    return heft_rank_sort(network, task_graph)


def hbmct_create_groups(network: Network, task_graph: TaskGraph) -> List[List[str]]:
    """Create Independent Groups for scheduling.

    Args:
        network (Network): The network.
        task_graph (TaskGraph): The task graph.

    Returns:
        List[List[str]]: List of groups.
    """
    rankings = hbmct_rank_sort(network, task_graph)
    groups: List[List[str]] = [[rankings[0]]]
    logging.debug("Upward: Rankings %s", rankings)
    for task_name in rankings[1:]:
        is_same_group = True
        for in_edge in task_graph.in_edges(task_name):
            if in_edge.source in groups[-1]:
                groups.append([task_name])
                is_same_group = False
                break
        if is_same_group:
            groups[-1].append(task_name)
    return groups


def calculate_est(
    network: Network,
    task_graph: TaskGraph,
    group: List[str],
    commtimes: Dict[Tuple[str, str], Dict[Tuple[str, str], float]],
    comp_schedule: Schedule,
    task_schedule: Dict[str, ScheduledTask],
) -> Dict[str, Dict[str, float]]:
    """Calculate the earliest start times for the given group on all nodes."""
    node_names = [node.name for node in network.nodes]
    est_table: Dict[str, Dict[str, float]] = {
        task_name: {node: 0.0 for node in node_names} for task_name in group
    }
    for task_name in group:
        for node_name in node_names:
            in_edges = task_graph.in_edges(task_name)
            max_arrival_time: float = max(
                [
                    0.0,
                    *[
                        task_schedule[in_edge.source].end
                        + (
                            commtimes[(task_schedule[in_edge.source].node, node_name)][
                                (in_edge.source, task_name)
                            ]
                        )
                        for in_edge in in_edges
                    ],
                ]
            )
            tasks = comp_schedule[node_name]
            if tasks:
                est_table[task_name][node_name] = max(max_arrival_time, tasks[-1].end)
            else:
                est_table[task_name][node_name] = max_arrival_time
    return est_table


def get_initial_assignments(
    network: Network,
    runtimes: Dict[str, Dict[str, float]],
    group: List[str],
    est_table: Dict[str, Dict[str, float]],
) -> Dict[str, List[str]]:
    """Get initial assignments of the tasks of an independent group based on their execution times."""
    node_names = [node.name for node in network.nodes]
    assignments: Dict[str, List[str]] = {node: [] for node in node_names}

    def avg_runtime(task_name: str) -> float:
        return float(np.mean([runtimes[node][task_name] for node in node_names]))

    for task_name in group:
        assigned_node = min(
            node_names, key=lambda n: est_table[task_name][n] + runtimes[n][task_name]
        )
        assignments[assigned_node].append(task_name)

    def est_sort_key(task_name: str, node_name: str) -> float:
        return est_table[task_name][node_name]

    for node_name in assignments:
        assignments[node_name].sort(key=lambda t: est_sort_key(t, node_name))
    return assignments


def get_ft(node_schedule: List[ScheduledTask]) -> float:
    """Calculate finish time for a node in a given schedule."""
    if node_schedule:
        return node_schedule[-1].end
    return 0


def get_ft_after_insert(
    new_task_name: str,
    node_name: str,
    assignments: List[str],
    node_schedule: List[ScheduledTask],
    est_table: Dict[str, Dict[str, float]],
    runtimes: Dict[str, Dict[str, float]],
    insert_position: int,
) -> Tuple[float, List[ScheduledTask], ScheduledTask]:
    """Calculate the finish time after inserting task in the schedule of a node."""
    new_assignments = assignments.copy()
    new_assignments.append(new_task_name)
    new_assignments.sort(key=lambda task: est_table[new_task_name][node_name])
    new_schedule = node_schedule.copy()
    new_task = None
    start_time = 0.0
    if new_schedule:
        start_time = max(start_time, node_schedule[-1].end)
    for task_id in reversed(range(insert_position, len(new_schedule))):
        del new_schedule[task_id]
    for task_name in new_assignments:
        start_time = est_table[task_name][node_name]
        if new_schedule:
            start_time = max(start_time, new_schedule[-1].end)
        task = ScheduledTask(
            node=node_name,
            name=task_name,
            start=start_time,
            end=start_time + runtimes[node_name][task_name],
        )
        new_schedule.append(task)
        if task_name == new_task_name:
            new_task = task
    if new_task is None:
        raise RuntimeError(
            "New task was not created in get_ft_after_insert."
        )  # Should not happen
    return new_schedule[-1].end, new_schedule, new_task


def delete_task_from_schedule(
    task_name: str,
    node_name: str,
    node_schedule: List[ScheduledTask],
    est_table: Dict[str, Dict[str, float]],
    runtimes: Dict[str, Dict[str, float]],
) -> Tuple[float, List[ScheduledTask]]:
    """Calculate the new schedule after removing task from a node schedule."""
    new_schedule = node_schedule.copy()
    i = None
    for i, task in enumerate(new_schedule):
        if task.name == task_name:
            del new_schedule[i]
            logging.debug("Deleted: %s", new_schedule)
            break
    if i is not None:
        for j in range(i, len(new_schedule)):
            task = new_schedule[j]
            start_time = est_table[task.name][node_name]
            if j != 0:
                start_time = max(start_time, new_schedule[j - 1].end)
            new_schedule[j] = ScheduledTask(
                node=node_name,
                name=task.name,
                start=start_time,
                end=start_time + runtimes[node_name][task.name],
            )

    new_ft = 0.0
    logging.debug(
        "Schedule of %s after removing %s : %s", node_name, task_name, new_schedule
    )
    if new_schedule:
        new_ft = new_schedule[-1].end
    return new_ft, new_schedule


class HbmctScheduler(Scheduler):
    """Schedules tasks using the HBMCT (Hybrid Minimum Completion Time) algorithm.

    Source: https://dx.doi.org/10.1137/0218016
    """

    @staticmethod
    def get_runtimes(
        network: Network, task_graph: TaskGraph
    ) -> Tuple[
        Dict[str, Dict[str, float]], Dict[Tuple[str, str], Dict[Tuple[str, str], float]]
    ]:
        """Get the expected runtimes of all tasks on all nodes."""
        runtimes: Dict[str, Dict[str, float]] = {}
        for node in network.nodes:
            runtimes[node.name] = {}
            for task in task_graph.tasks:
                runtimes[node.name][task.name] = task.cost / node.speed
                logging.debug(
                    "Task %s on node %s has runtime %s",
                    task.name,
                    node.name,
                    runtimes[node.name][task.name],
                )

        commtimes: Dict[Tuple[str, str], Dict[Tuple[str, str], float]] = {}
        for edge in network.edges:
            src, dst = edge.source, edge.target
            if (src, dst) not in commtimes:
                commtimes[src, dst] = {}
            if (dst, src) not in commtimes:
                commtimes[dst, src] = {}
            for dep in task_graph.dependencies:
                commtimes[src, dst][dep.source, dep.target] = dep.size / edge.speed
                commtimes[dst, src][dep.source, dep.target] = dep.size / edge.speed
                logging.debug(
                    "Task %s on node %s to task %s on node %s has communication time %s",
                    dep.source,
                    src,
                    dep.target,
                    dst,
                    commtimes[src, dst][dep.source, dep.target],
                )

        return runtimes, commtimes

    @staticmethod
    def schedule_groups(
        network: Network,
        task_graph: TaskGraph,
        groups: List[List[str]],
        runtimes: Dict[str, Dict[str, float]],
        commtimes: Dict[Tuple[str, str], Dict[Tuple[str, str], float]],
        comp_schedule: Schedule,
        task_schedule: Dict[str, ScheduledTask],
    ) -> Schedule:
        """Schedule all the groups independently."""
        node_names = [node.name for node in network.nodes]

        for group in groups:
            comp_group_start_positions = {
                node_name: len(comp_schedule[node_name]) for node_name in node_names
            }
            est_table = calculate_est(
                network, task_graph, group, commtimes, comp_schedule, task_schedule
            )
            average_est = {
                task_name: float(
                    np.mean([est for est in est_table[task_name].values()])
                )
                for task_name in est_table
            }
            assignments = get_initial_assignments(network, runtimes, group, est_table)

            # Build initial schedule for this group
            for node_name in assignments:
                for task_name in assignments[node_name]:
                    start_time = est_table[task_name][node_name]
                    tasks = comp_schedule[node_name]
                    if tasks:
                        start_time = max(start_time, tasks[-1].end)
                    task = ScheduledTask(
                        node=node_name,
                        name=task_name,
                        start=start_time,
                        end=start_time + runtimes[node_name][task_name],
                    )
                    comp_schedule.add_task(task)
                    task_schedule[task_name] = task

            logging.debug("Initial assignment for group %s: %s", group, comp_schedule)
            assignment_changed = True

            while assignment_changed:
                assignment_changed = False
                max_ft_node = max(
                    node_names, key=lambda x: get_ft(list(comp_schedule[x]))
                )
                tasks = comp_schedule[max_ft_node]
                if not tasks:
                    continue
                max_ft = tasks[-1].end
                avg_est_assignments = sorted(
                    assignments[max_ft_node],
                    key=lambda task_name: average_est[task_name],
                )
                logging.debug("current MFT %s with finish time %s", max_ft_node, max_ft)

                for task_name in avg_est_assignments:
                    logging.debug("Trying to move around task %s", task_name)
                    new_max_ft, max_ft_node_new_schedule = delete_task_from_schedule(
                        task_name,
                        max_ft_node,
                        list(comp_schedule[max_ft_node]),
                        est_table,
                        runtimes,
                    )
                    if new_max_ft < max_ft:
                        min_ft_node = None
                        min_ft_node_schedule = None
                        min_ft = float("inf")
                        for node_name in node_names:
                            if node_name != max_ft_node:
                                new_ft, node_schedule, _ = get_ft_after_insert(
                                    task_name,
                                    node_name,
                                    assignments[node_name],
                                    list(comp_schedule[node_name]),
                                    est_table,
                                    runtimes,
                                    comp_group_start_positions[node_name],
                                )
                                logging.debug(
                                    "EST of task %s on node %s: %s",
                                    task_name,
                                    node_name,
                                    est_table[task_name][node_name],
                                )
                                logging.debug(
                                    "New finish time for node %s after inserting task %s: %s",
                                    node_name,
                                    task_name,
                                    new_ft,
                                )
                                logging.debug(
                                    "Schedule of node %s after inserting %s : %s",
                                    node_name,
                                    task_name,
                                    node_schedule,
                                )
                                if new_ft < max_ft and new_ft < min_ft:
                                    min_ft = new_ft
                                    min_ft_node = node_name
                                    min_ft_node_schedule = node_schedule.copy()
                        if min_ft_node:
                            assignment_changed = True
                            # Update schedules - need to rebuild from the modified schedules
                            # Remove tasks from group start positions onwards
                            for n in node_names:
                                start_pos = comp_group_start_positions[n]
                                while len(comp_schedule[n]) > start_pos:
                                    comp_schedule.remove_task(comp_schedule[n][-1].name)

                            # Re-add tasks from new schedules
                            for t in max_ft_node_new_schedule[
                                comp_group_start_positions[max_ft_node] :
                            ]:
                                comp_schedule.add_task(t)
                                task_schedule[t.name] = t

                            if min_ft_node is None or min_ft_node_schedule is None:
                                raise RuntimeError(
                                    "Inconsistent state: min_ft_node or its schedule is None."
                                )  # Should not happen
                            for t in min_ft_node_schedule[
                                comp_group_start_positions[min_ft_node] :
                            ]:
                                comp_schedule.add_task(t)
                                task_schedule[t.name] = t

                            assignments[min_ft_node].append(task_name)
                            assignments[max_ft_node].remove(task_name)
                            logging.debug(
                                "New assignment for group %s: %s", group, comp_schedule
                            )
                            break

        return comp_schedule

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Computes the schedule for the task graph using the HBMCT algorithm."""
        comp_schedule = Schedule(task_graph, network)
        task_schedule: Dict[str, ScheduledTask] = {}

        if schedule is not None:
            comp_schedule = schedule.model_copy()
            task_schedule = {t.name: t for _, tasks in schedule.items() for t in tasks}

        runtimes, commtimes = HbmctScheduler.get_runtimes(network, task_graph)
        groups = hbmct_create_groups(network, task_graph)
        return self.schedule_groups(
            network,
            task_graph,
            groups,
            runtimes,
            commtimes,
            comp_schedule,
            task_schedule,
        )
