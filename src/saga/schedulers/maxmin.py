from functools import lru_cache
from typing import Dict, Optional

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph


class MaxMinScheduler(Scheduler):
    """Max-Min scheduler"""

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
            min_start_time (float): Minimum start time for tasks. Defaults to 0.0.

        Returns:
            Schedule: The schedule.
        """
        comp_schedule = Schedule(task_graph, network)
        scheduled_tasks: Dict[str, ScheduledTask] = {}

        if schedule is not None:
            comp_schedule = schedule.model_copy()
            scheduled_tasks = {
                task.name: task for _, tasks in schedule.items() for task in tasks
            }

        @lru_cache(maxsize=None)
        def get_eet(task_name: str, node_name: str) -> float:
            task = task_graph.get_task(task_name)
            node = network.get_node(node_name)
            return task.cost / node.speed

        @lru_cache(maxsize=None)
        def get_commtime(task1: str, task2: str, node1: str, node2: str) -> float:
            dep = task_graph.get_dependency(task1, task2)
            edge = network.get_edge(node1, node2)
            return dep.size / edge.speed

        @lru_cache(
            maxsize=None
        )  # Must clear cache after each iteration since schedule changes
        def get_eat(node_name: str) -> float:
            tasks = comp_schedule[node_name]
            return tasks[-1].end if tasks else min_start_time

        @lru_cache(
            maxsize=None
        )  # Must clear cache after each iteration since schedule changes
        def get_fat(task_name: str, node_name: str) -> float:
            in_edges = task_graph.in_edges(task_name)
            if not in_edges:
                return min_start_time
            return max(
                [
                    scheduled_tasks[in_edge.source].end
                    + get_commtime(
                        in_edge.source,
                        task_name,
                        scheduled_tasks[in_edge.source].node,
                        node_name,
                    )
                    for in_edge in in_edges
                ]
            )

        @lru_cache(
            maxsize=None
        )  # Must clear cache after each iteration since schedule changes
        def get_ect(task_name: str, node_name: str) -> float:
            return get_eet(task_name, node_name) + max(
                get_eat(node_name), get_fat(task_name, node_name)
            )

        def clear_caches():
            """Clear all caches."""
            get_eat.cache_clear()
            get_fat.cache_clear()
            get_ect.cache_clear()

        num_tasks = len(list(task_graph.tasks))
        node_names = [node.name for node in network.nodes]

        while len(scheduled_tasks) < num_tasks:
            available_tasks = [
                task.name
                for task in task_graph.tasks
                if task.name not in scheduled_tasks
                and all(
                    in_edge.source in scheduled_tasks
                    for in_edge in task_graph.in_edges(task.name)
                )
            ]
            # slight change from MinMin logic
            while available_tasks:
                min_ects = {
                    task_name: min(
                        get_ect(task_name, node_name) for node_name in node_names
                    )
                    for task_name in available_tasks
                }
                sched_task = max(min_ects, key=lambda task_name: min_ects[task_name])
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
                scheduled_tasks[sched_task] = new_task

                available_tasks.remove(sched_task)

                clear_caches()

        return comp_schedule
