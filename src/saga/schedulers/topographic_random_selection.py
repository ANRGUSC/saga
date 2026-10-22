from typing import Dict, Optional, Set
import random
import copy

from saga import (
    ConstraintViolation,
    Network,
    Schedule,
    Scheduler,
    ScheduledTask,
    TaskGraph,
    TaskGraphNode,
    TaskGraphEdge,
)


class TopographicRandomNodeScheduler(Scheduler):
    """Randomly schedules each task on a node it is allowed to run on, assigning the tasks in a random topographic order."""

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
        node_constraints: Optional[Dict[str, Set[str]]] = None,
    ) -> Schedule:
        """Schedule each task to a random node available to it.

        Args:
            network (Network): The network.
            task_graph (TaskGraph): The task graph.
            schedule (Optional[Schedule]): Optional initial schedule. Defaults to None.
            min_start_time (float): Minimum start time. Defaults to 0.0.
            node_constraints (Optional[Dict[str, Set[str]]]): Per-task placement
                constraints, applied only when a new schedule is constructed; a passed
                schedule uses its own constraints.

        Returns:
            Schedule: A schedule mapping nodes to a list of tasks.
        """
        if schedule is not None:
            comp_schedule = schedule.model_copy()
            scheduled_tasks = {
                t.name for _, tasks in comp_schedule.items() for t in tasks
            }
        else:
            comp_schedule = Schedule(
                task_graph, network, node_constraints=node_constraints
            )
            scheduled_tasks = set()

        for task in random_topographic_sort(task_graph):
            if task.name in scheduled_tasks:
                continue

            allowed = comp_schedule.allowed_nodes(task.name)
            candidates = [
                node
                for node in network.nodes
                if allowed is None or node.name in allowed
            ]
            if not candidates:
                raise ConstraintViolation(
                    f"Task {task.name} has no allowed node in the network "
                    f"(constraint: {sorted(allowed) if allowed else allowed})."
                )

            chosen = random.choice(candidates)

            start_time = comp_schedule.get_earliest_start_time(
                task, chosen, append_only=True, current_moment=min_start_time
            )
            comp_schedule.add_task(
                ScheduledTask(
                    node=chosen.name,
                    name=task.name,
                    start=start_time,
                    end=start_time + (task.cost / chosen.speed),
                )
            )
            scheduled_tasks.add(task.name)

        return comp_schedule


def random_topographic_sort(task_graph: TaskGraph) -> list[TaskGraphNode]:
    assignment_list: list[TaskGraphNode] = list()
    assigned_list: list[TaskGraphNode] = list()
    tasks: set[TaskGraphNode] = set(task_graph.tasks)
    dependencies: set[TaskGraphEdge] = set(task_graph.dependencies)
    frontier: set[TaskGraphNode] = set()  # create initial sets and lists
    for task in copy.deepcopy(
        tasks
    ):  # do initial sweep through main 'tasks' set for tasks w/out dependencies
        skip_task = False
        for edge in dependencies:
            if (
                edge.target == task.name
            ):  # iterate thru set of edges. if one has the task as a target, it's not one with no dependencies, skip it
                skip_task = True
                break
        if skip_task:
            skip_task = False
            continue
        else:
            assignment_list.append(
                task
            )  # if it has no dependencies, add to assignment_list for assignment
            tasks.remove(
                task
            )  # and remove it from the main set, as it is being assigned
    random.shuffle(assignment_list)  # randomization step
    for task in assignment_list:
        assigned_list.append(
            task
        )  # append randomized assignment_list to list that will be returned
        for edge in dependencies:
            if edge.source == task.name:
                frontier.add(
                    task_graph.get_task(edge.target)
                )  # add children of an assigned task to the frontier set
    assignment_list.clear()  # reset assignment_list for next slate of tasks
    while tasks:  # then while the main set is not empty:
        for task in copy.deepcopy(frontier):
            skip_task = False
            for edge in dependencies:
                if (
                    edge.target == task.name
                    and task_graph.get_task(edge.source) in tasks
                ):  # loop thru tasks in fontier and ask "are *any* of its parents in the main set"
                    frontier.remove(
                        task
                    )  # if yes, it has unassigned parents. remove it from the frontier set and move on to next task
                    skip_task = True
                    break
            if skip_task:
                skip_task = False
                continue
            else:
                assignment_list.append(
                    task
                )  # otherwise, add the task to the holding list for assignment
                frontier.remove(task)  # and remove from frontier/main set
                tasks.remove(task)
        random.shuffle(
            assignment_list
        )  # randomize list once all tasks removed from frontier
        for task in assignment_list:  # as before,
            assigned_list.append(
                task
            )  # append tasks from assignment_list to assigned_list
            for edge in dependencies:  # add any appened tasks' children to frontier
                if edge.source == task.name:
                    frontier.add(task_graph.get_task(edge.target))
        assignment_list.clear()  # reset assignment_list for next loop
    return assigned_list  # once main set is empty, all tasks assigned. return assigned_list
