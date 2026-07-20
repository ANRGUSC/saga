import pathlib
from typing import List, Optional

from saga import (
    Schedule,
    Scheduler,
    ScheduledTask,
    TaskGraph,
    Network,
    TaskGraphNode,
)
from saga.schedulers.heft import heft_rank_sort

thisdir = pathlib.Path(__file__).resolve().parent
"""
Adapted from: 
Atul Vikas Lakra, Dharmendra Kumar Yadav,
Multi-Objective Tasks Scheduling Algorithm for Cloud Computing Throughput Optimization,
ISSN 1877-0509,
https://doi.org/10.1016/j.procs.2015.04.158
"""


class MultiObjScheduler(Scheduler):
    """Schedules tasks using a multi-objective optimization approach."""

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (Optional[Dict[Hashable, List[Task]]], optional): The schedule. Defaults to None.
            min_start_time (float, optional): The minimum start time. Defaults to 0.0.

        Returns:
            Dict[str, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """

        rank_order = heft_rank_sort(network=network, task_graph=task_graph)
        rankings = {name: i for i, name in enumerate(rank_order)}
        schedule = schedule if schedule is not None else Schedule(task_graph, network)
        remaining_tasks = set(task_graph.tasks)
        network_nodes = sorted(network.nodes, key=lambda n: n.speed, reverse=True)
        current_node = 0
        ready_tasks = [t for t in task_graph.tasks if not task_graph.in_edges(t)]
        while remaining_tasks:
            non_dominated_tasks: List[TaskGraphNode] = []
            dominated_tasks: List[TaskGraphNode] = [Task for Task in ready_tasks]
            non_dominated_tasks.append(dominated_tasks.pop(0))

            for dom_task in list(dominated_tasks):
                to_remove = []
                is_dominated = False
                for ndom_task in non_dominated_tasks:
                    if self.dominates(ndom_task, dom_task, rankings):
                        is_dominated = True
                        break
                    elif self.dominates(dom_task, ndom_task, rankings):
                        # non_dominated_tasks.append(dom_task)
                        to_remove.append(ndom_task)
                if not is_dominated:
                    for task in to_remove:
                        non_dominated_tasks.remove(task)
                        dominated_tasks.append(task)
                    dominated_tasks.remove(dom_task)
                    non_dominated_tasks.append(dom_task)

            # scheduling tasks
            network_nodes = sorted(network.nodes, key=lambda n: n.speed, reverse=True)
            # sorting tasks
            sorted_tasks = sorted(
                non_dominated_tasks, key=lambda t: t.cost, reverse=True
            ) + sorted(dominated_tasks, key=lambda t: t.cost, reverse=True)

            task = sorted_tasks[0]
            selected_node = network_nodes[current_node]
            earliest_start_time = schedule.get_earliest_start_time(
                task,
                selected_node.name,
                current_moment=min_start_time,
            )
            if current_node < len(network_nodes) - 1:
                current_node += 1
            else:
                current_node = 0

            newtask = ScheduledTask(
                name=task.name,
                node=selected_node.name,
                start=earliest_start_time,
                end=earliest_start_time + task.cost / selected_node.speed,
            )
            schedule.add_task(newtask)
            ready_tasks.remove(task)
            remaining_tasks.remove(task)
            for edge in task_graph.out_edges(task):
                child = task_graph.get_task(edge.target)
                if child in remaining_tasks and all(
                    schedule.is_scheduled(e.source) for e in task_graph.in_edges(child)
                ):
                    ready_tasks.append(child)
        return schedule

    def dominates(
        self,
        task1: TaskGraphNode,
        task2: TaskGraphNode,
        rankings: dict,
    ) -> bool:
        """Check if task1 dominates task2.

        Args:
            task1 (TaskGraphNode): The first task.
            task2 (TaskGraphNode): The second task.
            rankings (dict): Map from task name to its HEFT rank order.
        Returns:
            bool: True if task1 dominates task2, False otherwise.
        """

        return (
            (task1.cost < task2.cost and rankings[task1.name] < rankings[task2.name])
            or (
                task1.cost <= task2.cost and rankings[task1.name] < rankings[task2.name]
            )
            or (
                task1.cost < task2.cost and rankings[task1.name] <= rankings[task2.name]
            )
        )
