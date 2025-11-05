from typing import Dict, Hashable, List, Set, Tuple
import networkx as nx
from saga.schedulers.parametric import (
    ParametricScheduler, IntialPriority, InsertTask
)

from saga.scheduler import Scheduler, ScheduledTask
from saga.utils.online_tools import schedule_estimate_to_actual, get_offline_instance
from saga.schedulers.parametric.components import (
    insert_funcs, initial_priority_funcs
)

class OnlineParametricScheduler(Scheduler):#need to add check for sufferage scheduler 
    def __init__(self, initial_priority: IntialPriority, insert_task: InsertTask) -> None: 
        self._parametric_scheduler = ParametricScheduler(
            initial_priority=initial_priority,
            insert_task=insert_task
        )

    def schedule_iterative(self,
                           network: nx.Graph,
                           task_graph: nx.DiGraph) -> Tuple[List[Dict[Hashable, List[ScheduledTask]]], List[Dict[Hashable, List[ScheduledTask]]], List[Dict[Hashable, List[ScheduledTask]]]]:
        """Online scheduling algorithm that produces a schedule, then waits for
        the next task to finish before producing the next schedule.

        Args:
            network (nx.Graph): The network graph where tasks will be scheduled.
            task_graph (nx.DiGraph): The directed task graph containing tasks and their dependencies.

        Returns:
            Tuple[List[Dict[Hashable, List[Task]]], List[Dict[Hashable, List[Task]]], List[Dict[Hashable, List[Task]]]]:
                A tuple containing three lists:
                - schedules_estimate: The estimated schedules based on the parametric scheduler.
                - schedules_hypothetical: The hypothetical schedules based on the estimates.
                - schedules_actual: The actual schedules after all tasks have been scheduled.
        """
        schedules_estimate: List[Dict[Hashable, List[ScheduledTask]]] = []
        schedules_hypothetical: List[Dict[Hashable, List[ScheduledTask]]] = []
        schedules_actual: List[Dict[Hashable, List[ScheduledTask]]] = []
        schedule_actual: Dict[Hashable, List[ScheduledTask]] = {
            node: [] for node in network.nodes
        }
        tasks_actual: Dict[str, ScheduledTask] = {}
        finished_tasks: Set[str] = set()
        current_time = 0

        while len(tasks_actual) < len(task_graph.nodes):
            # Generate a schedule estimate using the parametric scheduler
            schedule_estimate = self._parametric_scheduler.schedule(
                network=network,
                task_graph=task_graph,
                schedule=schedule_actual,       # Passing in the already-scheduled tasks
                min_start_time=current_time     # Tasks shouldn't be scheduled before this time
            )
            schedules_estimate.append(schedule_estimate)
            schedule_actual_hypothetical = schedule_estimate_to_actual(
                network=network,
                task_graph=task_graph,
                schedule_estimate=schedule_estimate
            )
            schedules_hypothetical.append(schedule_actual_hypothetical)

            # Get the next task that will finish executing
            tasks: List[ScheduledTask] = sorted(
                [
                    task for node_tasks in schedule_actual_hypothetical.values()
                    for task in node_tasks
                ],
                key=lambda x: x.start
            )
            next_task: ScheduledTask = min(
                [task for task in tasks if task.name not in finished_tasks],
                key=lambda x: x.end
            )
            # Move the current time to the end of the next task
            current_time = next_task.end
            finished_tasks.add(next_task.name)

            # Add all tasks that start before or at the current time
            # to the schedule_actual. These can no longer be rescheduled.
            # For simplicity, we rebuilt the schedule_actual from scratch
            # every iteration.
            schedule_actual = {node: [] for node in network.nodes}
            tasks_actual = {}
            for task in tasks:
                if task.start <= current_time:
                    schedule_actual[task.node].append(task)
                    tasks_actual[task.name] = task
            schedules_actual.append(schedule_actual)

        return schedules_estimate, schedules_hypothetical, schedules_actual

    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> Dict[Hashable, List[ScheduledTask]]:
        """Online scheduling algorithm that produces a schedule, then waits for
        the next task to finish before producing the next schedule. This method
        returns only the final schedule after all tasks have been scheduled.

        Args:
            network (nx.Graph): The network graph where tasks will be scheduled.
            task_graph (nx.DiGraph): The directed task graph containing tasks and their dependencies.
        
        Returns:
            Dict[Hashable, List[Task]]: A dictionary mapping nodes to lists of tasks scheduled on those nodes.
        """
        schedules, _, _ = self.schedule_iterative(network, task_graph)
        # Return the last schedule in the list
        return schedules[-1]


schedulers: Dict[str, ParametricScheduler] = {}
for name, insert_func in insert_funcs.items():
    for intial_priority_name, initial_priority_func in initial_priority_funcs.items():
        schedulers[f"{name}_{intial_priority_name}"] = OnlineParametricScheduler(
            initial_priority=initial_priority_func,
            insert_task=insert_func
        )
