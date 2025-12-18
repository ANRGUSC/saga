from typing import Dict, Optional

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph


class SufferageScheduler(Scheduler):
    """Implements a sufferage scheduler.

    Source: https://dx.doi.org/10.1007/978-3-540-69277-5_7
    """

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Schedules the task graph on the network

        Args:
            network (Network): The network.
            task_graph (TaskGraph): The task graph.
            schedule (Optional[Schedule]): Optional initial schedule. Defaults to None.
            min_start_time (float): Minimum start time. Defaults to 0.0.

        Returns:
            Schedule: The schedule.
        """
        comp_schedule = Schedule(task_graph, network)
        task_map: Dict[str, ScheduledTask] = {}

        if schedule is not None:
            comp_schedule = schedule.model_copy()
            task_map = {t.name: t for _, tasks in schedule.items() for t in tasks}

        node_names = [node.name for node in network.nodes]

        def get_eet(task_name: str, node_name: str) -> float:
            """Estimated execution time of a task on a node"""
            task = task_graph.get_task(task_name)
            node = network.get_node(node_name)
            return task.cost / node.speed

        def get_commtime(task1: str, task2: str, node1: str, node2: str) -> float:
            """Communication time to send task1's output from node1 to task2's input on node2"""
            dep = task_graph.get_dependency(task1, task2)
            edge = network.get_edge(node1, node2)
            return dep.size / edge.speed

        def get_eat(node_name: str) -> float:
            """Earliest available time on a node"""
            tasks = comp_schedule[node_name]
            return tasks[-1].end if tasks else min_start_time

        def get_fat(task_name: str, node_name: str) -> float:
            """Get file availability time of a task on a node"""
            in_edges = task_graph.in_edges(task_name)
            if not in_edges:
                return min_start_time
            return max(
                task_map[in_edge.source].end
                + get_commtime(
                    in_edge.source, task_name, task_map[in_edge.source].node, node_name
                )
                for in_edge in in_edges
            )

        def get_ect(task_name: str, node_name: str) -> float:
            """Get estimated completion time of a task on a node"""
            return get_eet(task_name, node_name) + max(
                get_eat(node_name), get_fat(task_name, node_name)
            )

        num_tasks = len(list(task_graph.tasks))
        while len(task_map) < num_tasks:
            available_tasks = [
                task.name
                for task in task_graph.tasks
                if task.name not in task_map
                and all(
                    in_edge.source in task_map
                    for in_edge in task_graph.in_edges(task.name)
                )
            ]

            sufferages: Dict[str, float] = {}
            for task_name in available_tasks:
                ect_values = [get_ect(task_name, node_name) for node_name in node_names]
                first_ect = min(ect_values)
                ect_values.remove(first_ect)
                second_ect = min(ect_values) if ect_values else first_ect
                sufferages[task_name] = second_ect - first_ect

            sched_task = max(sufferages, key=lambda t: sufferages[t])
            sched_node = min(
                node_names, key=lambda node_name: get_ect(sched_task, node_name)
            )

            new_task = ScheduledTask(
                node=sched_node,
                name=sched_task,
                start=max(get_eat(sched_node), get_fat(sched_task, sched_node)),
                end=get_ect(sched_task, sched_node),
            )
            comp_schedule.add_task(new_task)
            task_map[sched_task] = new_task

        return comp_schedule
