from typing import Dict, Optional

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph


class OLBScheduler(Scheduler):
    """Opportunistic Load Balancing scheduler

    Source: https://doi.org/10.1006/jpdc.2000.1714
    Summary: "(OLB) assigns each task, in arbitrary order, to the next machine that is expected
        to be available, regardless of the task's expected execution time on that machine"
        (from source).
    """

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Schedule tasks on nodes using the OLB algorithm.

        Args:
            network (Network): Network.
            task_graph (TaskGraph): Task graph.
            schedule (Optional[Schedule]): Optional initial schedule. Defaults to None.
            min_start_time (float): Minimum start time. Defaults to 0.0.

        Returns:
            Schedule: Schedule of the tasks on the network.
        """
        comp_schedule = Schedule(task_graph, network)
        scheduled_tasks: Dict[str, ScheduledTask] = {}

        if schedule is not None:
            comp_schedule = schedule.model_copy()
            scheduled_tasks = {
                t.name: t for _, tasks in schedule.items() for t in tasks
            }

        node_names = [node.name for node in network.nodes]

        for task in task_graph.topological_sort():
            if task.name in scheduled_tasks:
                continue

            next_available_node = min(
                node_names,
                key=lambda node_name: comp_schedule[node_name][-1].end
                if comp_schedule[node_name]
                else min_start_time,
            )

            in_edges = task_graph.in_edges(task.name)
            times = [
                # time node is available
                comp_schedule[next_available_node][-1].end
                if comp_schedule[next_available_node]
                else min_start_time,
                *[
                    scheduled_tasks[in_edge.source].end
                    + (
                        in_edge.size
                        / network.get_edge(
                            scheduled_tasks[in_edge.source].node, next_available_node
                        ).speed
                    )
                    for in_edge in in_edges
                ],
            ]
            start_time = max(times)
            node = network.get_node(next_available_node)
            exec_time = task.cost / node.speed
            new_task = ScheduledTask(
                name=task.name,
                node=next_available_node,
                start=start_time,
                end=start_time + exec_time,
            )

            comp_schedule.add_task(new_task)
            scheduled_tasks[task.name] = new_task

        return comp_schedule
