from functools import lru_cache
import random
from typing import Dict, Optional, Tuple

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph


class WBAScheduler(Scheduler):
    """Workflow-Based Allocation (WBA) scheduler.

    Source: http://dx.doi.org/10.1109/CCGRID.2005.1558639
    """

    def __init__(self, alpha: float = 0.5) -> None:
        """Initializes the WBA scheduler.

        Args:
            alpha (float, optional): The alpha parameter. Defaults to 0.5.
        """
        super(WBAScheduler, self).__init__()
        self.alpha = alpha

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Returns the schedule of the given task graph on the given network.

        Args:
            network (Network): The network.
            task_graph (TaskGraph): The task graph.
            schedule (Optional[Schedule]): Optional initial schedule. Defaults to None.
            min_start_time (float): Minimum start time. Defaults to 0.0.

        Returns:
            Schedule: The schedule.
        """
        comp_schedule = Schedule(task_graph, network)
        scheduled_tasks: Dict[str, ScheduledTask] = {}

        if schedule is not None:
            comp_schedule = schedule.model_copy()
            scheduled_tasks = {
                t.name: t for _, tasks in schedule.items() for t in tasks
            }

        node_names = [node.name for node in network.nodes]

        @lru_cache(maxsize=None)
        def get_eet(task_name: str, node_name: str) -> float:
            """Estimated execution time of task on node"""
            task = task_graph.get_task(task_name)
            node = network.get_node(node_name)
            return task.cost / node.speed

        @lru_cache(maxsize=None)
        def get_commtime(task1: str, task2: str, node1: str, node2: str) -> float:
            """Communication time between task1 and task2 on node1 and node2"""
            dep = task_graph.get_dependency(task1, task2)
            edge = network.get_edge(node1, node2)
            return dep.size / edge.speed

        @lru_cache(
            maxsize=None
        )  # Must clear cache after each iteration since schedule changes
        def get_eat(node_name: str) -> float:
            """Earliest available time of node"""
            tasks = comp_schedule[node_name]
            return tasks[-1].end if tasks else min_start_time

        @lru_cache(
            maxsize=None
        )  # Must clear cache after each iteration since schedule changes
        def get_fat(task_name: str, node_name: str) -> float:
            """Latest available time of task on node"""
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

        @lru_cache(
            maxsize=None
        )  # Must clear cache after each iteration since schedule changes
        def get_est(task_name: str, node_name: str) -> float:
            """Earliest start time of task on node"""
            return max(get_eat(node_name), get_fat(task_name, node_name))

        @lru_cache(
            maxsize=None
        )  # Must clear cache after each iteration since schedule changes
        def get_ect(task_name: str, node_name: str) -> float:
            """Earliest completion time of task on node"""
            return get_eet(task_name, node_name) + get_est(task_name, node_name)

        def clear_caches():
            """Clear all caches."""
            get_eat.cache_clear()
            get_fat.cache_clear()
            get_est.cache_clear()
            get_ect.cache_clear()

        cur_makespan = min_start_time
        num_tasks = len(list(task_graph.tasks))
        while len(scheduled_tasks) < num_tasks:
            available_tasks = [
                task.name
                for task in task_graph.tasks
                if (
                    task.name not in scheduled_tasks
                    and all(
                        in_edge.source in scheduled_tasks
                        for in_edge in task_graph.in_edges(task.name)
                    )
                )
            ]

            while available_tasks:
                makespan_increases: Dict[Tuple[str, str], float] = {}
                for task_name in available_tasks:
                    for node_name in node_names:
                        makespan_increases[task_name, node_name] = max(
                            get_ect(task_name, node_name) - cur_makespan, 0
                        )

                i_min = min(makespan_increases.values())
                i_max = max(makespan_increases.values())
                avail_pairs = [
                    key
                    for key, value in makespan_increases.items()
                    if value <= i_min + self.alpha * (i_max - i_min)
                ]

                sched_task, sched_node = random.choice(avail_pairs)
                new_task = ScheduledTask(
                    node=sched_node,
                    name=sched_task,
                    start=get_est(sched_task, sched_node),
                    end=get_ect(sched_task, sched_node),
                )
                comp_schedule.add_task(new_task)
                scheduled_tasks[sched_task] = new_task
                cur_makespan = max(cur_makespan, new_task.end)
                available_tasks.remove(sched_task)

                clear_caches()

        return comp_schedule
