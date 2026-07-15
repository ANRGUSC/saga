from typing import Dict, Optional, Set, Tuple
import numpy as np

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph


class ETFScheduler(Scheduler):
    """Earliest Task First scheduler"""

    def _get_start_times(
        self,
        task_map: Dict[str, ScheduledTask],
        ready_tasks: Set[str],
        ready_nodes: Set[str],
        task_graph: TaskGraph,
        network: Network,
        min_start_time: float = 0.0,
    ) -> Dict[str, Tuple[str, float]]:
        """Returns the earliest possible start times of the ready tasks on the ready nodes

        Args:
            task_map (Dict[str, ScheduledTask]): The scheduled tasks.
            ready_tasks (Set[str]): The ready tasks.
            ready_nodes (Set[str]): The ready nodes.
            task_graph (TaskGraph): The task graph.
            network (Network): The network.
            min_start_time (float): Minimum start time. Defaults to 0.0.

        Returns:
            Dict[str, Tuple[str, float]]: The start times of the ready tasks on the ready nodes.
        """
        start_times: Dict[str, Tuple[str, float]] = {}
        for task_name in ready_tasks:
            min_node = ""  # arbitrary initialization
            mini_min_start_time = np.inf
            for node_name in ready_nodes:
                in_edges = task_graph.in_edges(task_name)
                max_arrival_time = max(
                    [
                        min_start_time,
                        *[
                            task_map[in_edge.source].end
                            + (
                                in_edge.size
                                / network.get_edge(
                                    task_map[in_edge.source].node, node_name
                                ).speed
                            )
                            for in_edge in in_edges
                        ],
                    ]
                )
                if max_arrival_time < mini_min_start_time:
                    mini_min_start_time = max_arrival_time
                    min_node = node_name
            start_times[task_name] = min_node, mini_min_start_time
        return start_times

    def _get_ready_tasks(
        self, task_map: Dict[str, ScheduledTask], task_graph: TaskGraph
    ) -> Set[str]:
        """Returns the ready tasks

        Args:
            task_map (Dict[str, ScheduledTask]): The scheduled tasks.
            task_graph (TaskGraph): The task graph.

        Returns:
            Set[str]: The ready tasks.
        """
        return {
            task.name
            for task in task_graph.tasks
            if task.name not in task_map
            and all(
                in_edge.source in task_map for in_edge in task_graph.in_edges(task.name)
            )
        }

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Returns the best schedule (minimizing makespan) for a problem instance using ETF

        Args:
            network: Network
            task_graph: Task graph
            schedule: Optional initial schedule. Defaults to None.
            min_start_time: Minimum start time. Defaults to 0.0.

        Returns:
            A Schedule object containing the computed schedule.
        """
        current_moment = min_start_time
        next_moment = np.inf

        comp_schedule = Schedule(task_graph, network)
        task_map: Dict[str, ScheduledTask] = {}

        if schedule is not None:
            comp_schedule = schedule.model_copy()
            task_map = {t.name: t for _, tasks in schedule.items() for t in tasks}
            if task_map:
                current_moment = min_start_time
                next_moment = min(
                    (
                        t.end
                        for _, tasks in comp_schedule.items()
                        for t in tasks
                        if t.end > current_moment
                    ),
                    default=np.inf,
                )
            else:
                current_moment = min_start_time
                next_moment = np.inf

        node_names = {node.name for node in network.nodes}
        num_tasks = len(list(task_graph.tasks))

        while len(task_map) < num_tasks:
            ready_tasks = self._get_ready_tasks(task_map, task_graph)
            ready_nodes = {
                node_name
                for node_name in node_names
                if not comp_schedule[node_name]
                or comp_schedule[node_name][-1].end <= current_moment
            }
            while ready_tasks and ready_nodes:
                start_times = self._get_start_times(
                    task_map,
                    ready_tasks,
                    ready_nodes,
                    task_graph,
                    network,
                    min_start_time=current_moment,
                )
                task_to_schedule = min(
                    list(start_times.keys()),
                    key=lambda task_name: start_times[task_name][1],
                )
                node_to_schedule_on, start_time = start_times[task_to_schedule]

                start_time = max(start_time, current_moment)

                if start_time <= next_moment:
                    task = task_graph.get_task(task_to_schedule)
                    node = network.get_node(node_to_schedule_on)
                    new_task = ScheduledTask(
                        node=node_to_schedule_on,
                        name=task_to_schedule,
                        start=start_time,
                        end=start_time + (task.cost / node.speed),
                    )
                    comp_schedule.add_task(new_task)
                    task_map[task_to_schedule] = new_task
                    ready_tasks.remove(task_to_schedule)
                    ready_nodes.remove(node_to_schedule_on)
                    if new_task.end < next_moment:
                        next_moment = new_task.end
                else:
                    break

            current_moment = next_moment
            next_moment = min(
                [
                    np.inf,
                    *[
                        task.end
                        for task in task_map.values()
                        if task.end > current_moment
                    ],
                ]
            )

        return comp_schedule
