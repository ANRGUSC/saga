from typing import Dict, List, Optional

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph


class MaxTPScheduler(Scheduler):
    """Max-Throughput scheduler.

    Greedily schedules the largest available (ready) task on the node that
    maximizes the resulting schedule's throughput, i.e. minimizes the
    bottleneck (busiest compute node or comm link) the schedule would have
    if that task were added.
    """

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Schedules the task graph on the network.

        Args:
            network (Network): The network.
            task_graph (TaskGraph): The task graph.
            schedule (Optional[Schedule]): Optional initial schedule. Defaults to None.
            min_start_time (float): Minimum start time for tasks. Defaults to 0.0.

        Returns:
            Schedule: The schedule.
        """
        comp_schedule = Schedule(task_graph, network)
        scheduled: Dict[str, ScheduledTask] = {}

        if schedule is not None:
            comp_schedule = schedule.model_copy()
            scheduled = {
                task.name: task for _, tasks in schedule.items() for task in tasks
            }

        # Sort so tie-breaking is PYTHONHASHSEED-independent (network.nodes is a frozenset).
        node_names = sorted(node.name for node in network.nodes)
        num_tasks = len(list(task_graph.tasks))

        while len(scheduled) < num_tasks:
            available_tasks = sorted(
                (
                    task
                    for task in task_graph.tasks
                    if task.name not in scheduled
                    and all(
                        in_edge.source in scheduled
                        for in_edge in task_graph.in_edges(task.name)
                    )
                ),
                key=lambda task: task.name,
            )

            largest_task = max(available_tasks, key=lambda task: task.cost)

            best_task: Optional[ScheduledTask] = None
            for node_name in node_names:
                node = network.get_node(node_name)
                start_time = comp_schedule.get_earliest_start_time(
                    task=largest_task,
                    node=node,
                    current_moment=min_start_time,
                )
                candidate = ScheduledTask(
                    node=node_name,
                    name=largest_task.name,
                    start=start_time,
                    end=start_time + largest_task.cost / node.speed,
                )
                if best_task is None or comp_schedule.bottleneck_if_added(
                    candidate
                ) < comp_schedule.bottleneck_if_added(best_task):
                    best_task = candidate

            comp_schedule.add_task(best_task)
            scheduled[best_task.name] = best_task

        return comp_schedule
