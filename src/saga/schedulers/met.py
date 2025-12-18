from functools import partial
from typing import Dict, Optional

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph


class METScheduler(Scheduler):
    """Minimum Execution Time scheduler"""

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Returns the schedule of the tasks on the network

        Args:
            network (Network): The network.
            task_graph (TaskGraph): The task graph.
            schedule (Optional[Schedule]): Optional initial schedule. Defaults to None.
            min_start_time (float): Minimum start time. Defaults to 0.0.

        Returns:
            Schedule: The schedule of the tasks on the network.
        """
        comp_schedule = Schedule(task_graph, network)
        scheduled_tasks: Dict[str, ScheduledTask] = {}

        if schedule is not None:
            comp_schedule = schedule.model_copy()
            scheduled_tasks = {
                t.name: t for _, tasks in schedule.items() for t in tasks
            }

        def get_exec_time(task_name: str, node_name: str) -> float:
            task = task_graph.get_task(task_name)
            node = network.get_node(node_name)
            return task.cost / node.speed

        def get_commtime(task1: str, task2: str, node1: str, node2: str) -> float:
            dep = task_graph.get_dependency(task1, task2)
            edge = network.get_edge(node1, node2)
            return dep.size / edge.speed

        def get_eat(node_name: str) -> float:
            tasks = comp_schedule[node_name]
            return tasks[-1].end if tasks else min_start_time

        def get_fat(task_name: str, node_name: str) -> float:
            in_edges = task_graph.in_edges(task_name)
            if not in_edges:
                return min_start_time
            return max(
                scheduled_tasks[in_edge.source].end
                + get_commtime(
                    in_edge.source,
                    task_name,
                    scheduled_tasks[in_edge.source].node,
                    node_name,
                )
                for in_edge in in_edges
            )

        node_names = [node.name for node in network.nodes]

        for task in task_graph.topological_sort():
            if task.name in scheduled_tasks:
                continue

            # Find node with minimum execution time for the task
            sched_node = min(node_names, key=partial(get_exec_time, task.name))

            start_time = max(get_eat(sched_node), get_fat(task.name, sched_node))
            end_time = start_time + get_exec_time(task.name, sched_node)

            # Add task to the schedule
            new_task = ScheduledTask(
                node=sched_node, name=task.name, start=start_time, end=end_time
            )
            comp_schedule.add_task(new_task)
            scheduled_tasks[task.name] = new_task

        return comp_schedule
