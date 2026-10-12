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
    TaskGraphEdge
)


class TopographicRandomNodeScheduler(Scheduler):
    """Randomly schedules each task on a node it is allowed to run on, assigning the tasks in a random topographic order.
    """

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
    assignment_list = list()
    assigned_list = list()
    tasks = set(task_graph.tasks)
    dependencies = set(task_graph.dependencies)
    frontier = set()
    for task in copy.deepcopy(tasks):
        skip_task = False
        for edge in dependencies:
            if edge.target == task.name:
                skip_task = True
                break
        if skip_task:
            skip_task = False
            continue
        else:
            assignment_list.append(task)
            tasks.remove(task)
    assignment_list = random.shuffle(assignment_list)
    for task in assignment_list:
        assigned_list.append(task)
        for edge in dependencies:
            if edge.source == task.name:
                frontier.add(task_graph.get_task(edge.target))
    assignment_list.clear()
    while tasks:
        for task in frontier:
            skip_task = False
            for edge in dependencies:
                if edge.target == task.name and task_graph.get_task(edge.source) in tasks:
                    frontier.remove(task)
                    skip_task = True
                    break
            if skip_task:
                skip_task = False
                continue
            else:
                assignment_list.append(task)
                frontier.remove(task)
                tasks.remove(task)
        assignment_list = random.shuffle(assignment_list)
        for task in assignment_list:
            assigned_list.append(task)
            for edge in dependencies:
                if edge.source == task.name:
                    frontier.add(task_graph.get_task(edge.target))
    return assigned_list
            

        


    