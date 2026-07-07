from typing import Dict, Optional, Set

from saga import (
    ConstraintViolation,
    Network,
    Schedule,
    Scheduler,
    ScheduledTask,
    TaskGraph,
)


class FastestNodeScheduler(Scheduler):
    """Schedules each task on the fastest node it is allowed to run on.

    With no placement constraints this is the classic baseline: every task lands on the
    single fastest node in the network. When the schedule carries per-task constraints,
    each task instead takes the fastest node in its own allowed set, so tasks pinned to a
    slower tier (edge or fog) stay there while the rest still race to the fastest node.
    """

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
        node_constraints: Optional[Dict[str, Set[str]]] = None,
    ) -> Schedule:
        """Schedule each task on the fastest node available to it.

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
            comp_schedule = Schedule(task_graph, network, node_constraints=node_constraints)
            scheduled_tasks = set()

        for task in task_graph.topological_sort():
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
            chosen = max(candidates, key=lambda node: node.speed)

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
