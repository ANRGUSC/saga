from typing import Dict, Optional

import random
import math
from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph, ConstraintViolation
from saga.schedulers.parametric.components import (
    RandomTopological,
)

START_TEMP: int = 300
seed: int = 1152345234234  ##TODO: make this a real seed

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

        comp_schedule = sort_to_tasks(initial_sort, scheduled_tasks, comp_schedule, network, min_start_time)
        ## get makespan of this schedule
        best_schedule = comp_schedule
        best_makespan = comp_schedule.makespan()
        ## perterb the topographic sort
        temp = START_TEMP
        previous_sort = get_perterbed_sort(initial_sort)
        while temp > 0:
            new_sort = get_perterbed_sort(previous_sort)
        ## produce new schedule, get makespan
            new_schedule = Schedule(
                task_graph, network, node_constraints=node_constraints
            )
            new_schedule = sort_to_tasks(new_sort, set(), new_schedule, network, min_start_time)
            new_makespan = new_schedule.makespan()
        ## compare, accept/reject, cool
            delta_makespan = new_makespan - best_makespan
            if delta_makespan > 0:
                best_makespan = new_makespan
                best_schedule = new_schedule
                previous_sort = new_sort
            else:
                prob = math.exp(-delta_makespan/temp)
                if random.random() > prob:
                    previous_sort = new_sort
            temp -= 1
        return best_schedule
## return to perterb step, exit when iterations are exhausted, return schedule with best makespan



##Helper functions
def sort_to_tasks(sort, scheduled_tasks, schedule, network, min_start_time):
    """Helper function that assigns a valid topological sort to the tasks of a task graph. Utilizes
    a greedy fastest node first method."""
    for task in sort:
        if task.name in scheduled_tasks:
            continue

        allowed = schedule.allowed_nodes(task.name)
        candidates = [
            node
            for node in network.nodes
            if allowed is None or node.name in allowed
        ]
        if not candidates:
            raise ConstraintViolation(
                f"Task {task.name} has no allowed node in the network "
                f"(constraint: {sorted(allowed) if allowed else allowed})."
            )
        chosen = max(candidates, key=lambda node: node.speed)

        start_time = schedule.get_earliest_start_time(
            task, chosen, append_only=True, current_moment=min_start_time
        )
        schedule.add_task(
            ScheduledTask(
                node=chosen.name,
                name=task.name,
                start=start_time,
                end=start_time + (task.cost / chosen.speed),
            )
        )
        scheduled_tasks.add(task.name)
    return schedule

def get_perterbed_sort(sort: list[str], task_graph: TaskGraph) -> list[str]:
    """Helper function that selects a task at a random index in a valid topological sort and moves it to a random
    different index in the sort such that the new list is still a valid topological sort. Returns a list of tasks."""
    index = random.randint(0, len(sort)-1)
    candidate_task = sort[index]
    parent_tasks = set()
    child_tasks = set()
    for edge in task_graph.dependencies:
        if edge.source == candidate_task:
            child_tasks.add(edge.target)
        elif edge.target == candidate_task:
            parent_tasks.add(edge.source)
    best_index_left = 0
    for task in parent_tasks:
        if sort.index(task) > best_index_left:
            best_index_left = sort.index(task)
    best_index_right = len(sort)-1
    for task in child_tasks:
        if sort.index(task) < best_index_right:
            best_index_right = sort.index(task)
    new_index = random.randint(best_index_left+1, best_index_right-1)
    sort.insert(new_index, sort.pop(index))
    return sort

