from typing import Dict, Optional

import random
import math
import numpy as np
from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph, ConstraintViolation
from saga.schedulers.parametric.components import (
    RandomTopological,
)

START_TEMP: int = 300
seed: int = 1152345234234 ##TODO: make this a real seed

random_sorter = RandomTopological(seed=seed)

class Parameter_Annealing(Scheduler):
    """Scheduler that uses simulated annealing with a neighbor function that perterbs the 
    order of the topographic sort."""
    def schedule(
            self,
            network: Network,
            task_graph: TaskGraph,
            schedule: Optional[Schedule] = None,
            min_start_time: float = 0.0,
            node_constraints: Optional[Dict[str, Set[str]]] = None,
        ) -> Schedule:
        if schedule is not None:
            comp_schedule = schedule.model_copy()
            scheduled_tasks = {
                t.name for _, tasks in comp_schedule.items() for t in tasks
            }
        else:
            comp_schedule = Schedule(
                task_graph, network, node_constraints=node_constraints
            )
        ## initial random distribution of tasks to nodes, done by a random topographic sort of tasks
        initial_sort = random_sorter.call(network, task_graph)
        scheduled_tasks = set()

        comp_schedule = sort_to_tasks(initial_sort, scheduled_tasks, task_graph, comp_schedule, network, min_start_time)
        ## get makespan of this schedule
        best_schedule = comp_schedule
        best_makespan = comp_schedule.makespan
        ## perterb the topographic sort
        temp = START_TEMP
        previous_sort = initial_sort
        current_makespan = best_makespan
        while temp > 0:
            new_sort = get_perterbed_sort(previous_sort, task_graph)
        ## produce new schedule, get makespan
            new_schedule = Schedule(
                task_graph, network, node_constraints=node_constraints
            )
            new_schedule = sort_to_tasks(new_sort, set(), task_graph, new_schedule, network, min_start_time)
            new_makespan = new_schedule.makespan
        ## compare, accept/reject, cool
            delta = new_makespan - current_makespan   # positive = worse
            if delta < 0 or random.random() < math.exp(-delta / temp):
                previous_sort, current_makespan = new_sort, new_makespan
            if new_makespan < best_makespan:
                best_makespan, best_schedule = new_makespan, new_schedule
            temp -= 1
        return best_schedule
## return to perterb step, exit when iterations are exhausted, return schedule with best makespan



##Helper functions
def sort_to_tasks(sort:list[str], scheduled_tasks:set[ScheduledTask], task_graph: TaskGraph, schedule:Schedule, network:Network, min_start_time:float)->Schedule:
    """Helper function that assigns a valid topological sort to the tasks of a task graph. Utilizes the same greedy method as HEFT."""
    for task in sort:
        task_node = task_graph.get_task(task)
        if task_node.name in scheduled_tasks:
            continue

        allowed = schedule.allowed_nodes(task_node.name)
        candidates = [
            node
            for node in network.nodes
            if allowed is None or node.name in allowed
        ]
        if not candidates:
            raise ConstraintViolation(
                f"Task {task_node.name} has no allowed node in the network "
                f"(constraint: {sorted(allowed) if allowed else allowed})."
            )
        if schedule.is_scheduled(task):
            continue
        min_finish_time = np.inf
        best_node = next(iter(network.nodes))
        for node in candidates:
            start_time = schedule.get_earliest_start_time(
                task=task, 
                node=node, 
                append_only=False, 
                current_moment=min_start_time
            )
            runtime = task_graph.get_task(task).cost / network.get_node(node.name).speed
            finish_time = start_time + runtime

            if finish_time < min_finish_time:
                min_finish_time = finish_time
                best_node = node

        new_task = ScheduledTask(
            node=best_node.name,
            name=task,
            start=min_finish_time - (task_graph.get_task(task).cost / network.get_node(best_node).speed),
            end=min_finish_time
        )
        schedule.add_task(new_task)
    return schedule

def get_perterbed_sort(sort: list[str], task_graph: TaskGraph) -> list[str]:
    """Helper function that selects a task at a random index in a valid topological sort and moves it to a random
    different index in the sort such that the new list is still a valid topological sort. Returns a list of tasks."""
    sort = list(sort)
    index = random.randint(0, len(sort)-1)
    candidate_task = sort[index]
    parent_tasks = set()
    child_tasks = set()
    for edge in task_graph.dependencies:
        if edge.source == candidate_task:
            child_tasks.add(edge.target)
        elif edge.target == candidate_task:
            parent_tasks.add(edge.source)
    best_index_left = -1
    for task in parent_tasks:
        if sort.index(task) > best_index_left:
            best_index_left = sort.index(task)
    best_index_right = len(sort)
    for task in child_tasks:
        if sort.index(task) < best_index_right:
            best_index_right = sort.index(task)
    new_index = random.randint(best_index_left+1, best_index_right-1)
    sort.insert(new_index, sort.pop(index))
    return sort

