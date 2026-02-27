from typing import Dict, Optional

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph
from saga.constraints import Constraints


class FastestNodeScheduler(Scheduler):
    """Schedules all tasks on the node with the highest processing speed"""

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Schedules all tasks on the node with the highest processing speed

        Args:
            network (Network): The network.
            task_graph (TaskGraph): The task graph.
            schedule (Optional[Schedule]): Optional initial schedule. Defaults to None.
            min_start_time (float): Minimum start time. Defaults to 0.0.

        Returns:
            Schedule: A schedule mapping nodes to a list of tasks.
        """
        placement_constraints = Constraints.from_task_graph(task_graph)
        node_names = [node.name for node in network.nodes]

        comp_schedule = Schedule(task_graph, network)
        scheduled_tasks: Dict[str, ScheduledTask] = {}

        if schedule is not None:
            comp_schedule = schedule.model_copy()
            scheduled_tasks = {
                t.name: t for _, tasks in schedule.items() for t in tasks
            }

        for task in task_graph.topological_sort():
            if task.name in scheduled_tasks:
                continue

            candidate_names = placement_constraints.get_candidate_nodes(task.name, node_names)
            fastest_node = max(
                (network.get_node(n) for n in candidate_names),
                key=lambda node: node.speed,
            )
            exec_time = task.cost / fastest_node.speed

            node_free_time = min_start_time
            if comp_schedule[fastest_node.name]:
                node_free_time = comp_schedule[fastest_node.name][-1].end

            # For most instances, the data should probably arrive immediately
            # since everything is executing on the same node.
            data_arrival_time = min_start_time
            in_edges = task_graph.in_edges(task.name)
            if in_edges:
                data_arrival_time = max(
                    scheduled_tasks[in_edge.source].end
                    + (
                        in_edge.size
                        / network.get_edge(
                            scheduled_tasks[in_edge.source].node, fastest_node.name
                        ).speed
                    )
                    for in_edge in in_edges
                )
            start_time = max(node_free_time, data_arrival_time)

            new_task = ScheduledTask(
                node=fastest_node.name,
                name=task.name,
                start=start_time,
                end=start_time + exec_time,
            )
            comp_schedule.add_task(new_task)
            scheduled_tasks[task.name] = new_task

        return comp_schedule
