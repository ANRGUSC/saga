from typing import Dict, Hashable, List, Set
import networkx as nx
from saga.schedulers.parametric import (
    ParametricScheduler, IntialPriority, InsertTask
)

from saga.scheduler import Scheduler, Task
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

    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """
        A 'live' scheduling loop using HEFT in a dynamic manner.
        We assume each node/task has 'weight_actual' and 'weight_estimate' attributes.
        
        1) No longer needed
        2) Calls the standard HEFT for an 'estimated' schedule.
        3) Converts that schedule to 'actual' times using schedule_estimate_to_actual.
        4) Commits the earliest-finishing new task to schedule_actual.
        5) Repeats until all tasks are scheduled.        

        comp_schedule = {
        "Node1": [Task(node="Node1", name="TaskA", start=0, end=5)],
        "Node2": [Task(node="Node2", name="TaskB", start=3, end=8)],
        """
        schedule_actual: Dict[Hashable, List[Task]] = {
            node: [] for node in network.nodes
        }
        tasks_actual: Dict[str, Task] = {}
        finished_tasks: Set[str] = set()
        current_time = 0

        while len(tasks_actual) < len(task_graph.nodes):
            # Generate a schedule estimate using the parametric scheduler
            schedule_actual_hypothetical = schedule_estimate_to_actual(
                network=network,
                task_graph=task_graph,
                schedule_estimate=self._parametric_scheduler.schedule(
                    network=network,
                    task_graph=task_graph,
                    schedule=schedule_actual,       # Passing in the already-scheduled tasks
                    min_start_time=current_time     # Tasks shouldn't be scheduled before this time
                )
            )

            # Get the next task that will finish executing
            tasks: List[Task] = sorted(
                [
                    task for node_tasks in schedule_actual_hypothetical.values()
                    for task in node_tasks
                ],
                key=lambda x: x.start
            )
            next_task: Task = min(
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

        return schedule_actual



schedulers: Dict[str, ParametricScheduler] = {}
for name, insert_func in insert_funcs.items():
    for intial_priority_name, initial_priority_func in initial_priority_funcs.items():
        schedulers[f"{name}_{intial_priority_name}"] = OnlineParametricScheduler(
            initial_priority=initial_priority_func,
            insert_task=insert_func
        )
