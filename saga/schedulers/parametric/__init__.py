from abc import ABC, abstractmethod
from copy import deepcopy
import logging
from typing import Any, Dict, Hashable, List, Optional, Tuple
import networkx as nx

from saga.scheduler import Scheduler, Task


class IntialPriority(ABC):
    @abstractmethod
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> List[Hashable]:
        """Return the initial priority of the tasks.
        
        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            List[Hashable]: The initial priority of the tasks.
        """
        pass

ScheduleType = Dict[Hashable, List[Task]]

class UpdatePriority(ABC):
    @abstractmethod
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: ScheduleType,
                 priority: List[Hashable]) -> List[Hashable]:
        """Return the updated priority of the tasks.
        
        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (ScheduleType): The schedule.
            priority (List[Hashable]): The current priority of the tasks.

        Returns:
            List[Hashable]: The updated priority of the tasks.
        """
        pass

class InsertTask(ABC):
    @abstractmethod
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: ScheduleType,
                 task: Hashable) -> Task:
        """Insert a task into the schedule.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (ScheduleType): The schedule.
            task (Hashable): The task.  

        Returns:
            Task: The inserted task
        """
        pass


class ParametricScheduler(Scheduler):
    def __init__(self,
                 initial_priority: IntialPriority,
                 update_priority: UpdatePriority,
                 insert_task: InsertTask) -> None:
            super().__init__()
            self.initial_priority = initial_priority
            self.update_priority = update_priority
            self.insert_task = insert_task
    
    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: Optional[ScheduleType] = None) -> ScheduleType:
        """Schedule the tasks on the network.
        
        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (Optional[ScheduleType]): The current schedule.

        Returns:
            Dict[Hashable, List[Task]]: A dictionary mapping nodes to a list of tasks executed on the node.
        """
        queue = self.initial_priority(network, task_graph)
        schedule = {node: [] for node in network.nodes} if schedule is None else deepcopy(schedule)
        scheduled_tasks: Dict[Hashable, Task] = {}
        while queue:
            queue = self.update_priority(network, task_graph, schedule, queue)
            task_name = queue.pop(0)
            task = self.insert_task(network, task_graph, schedule, task_name)
            logging.debug("Inserted task %s on node %s at time %s.", task.name, task.node, task.start)
            scheduled_tasks[task.name] = task
        return schedule

class ParametricKDepthScheduler(Scheduler):
    def __init__(self,
                 scheduler: ParametricScheduler,
                 k_depth: int) -> None:
            super().__init__()
            self.scheduler = scheduler
            self.k_depth = k_depth
    
    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: Optional[ScheduleType] = None) -> ScheduleType:
        """Schedule the tasks on the network.
        
        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (Optional[ScheduleType]): The current schedule.

        Returns:
            Dict[Hashable, List[Task]]: A dictionary mapping nodes to a list of tasks executed on the node.
        """
        schedule = {node: [] for node in network.nodes} if schedule is None else deepcopy(schedule)
        scheduled_tasks: Dict[Hashable, Task] = {
            task.name: task
            for node, tasks in schedule.items()
            for task in tasks
        }

        ready_tasks = {
            task for task in task_graph.nodes
            if all(pred in scheduled_tasks for pred in task_graph.predecessors(task))
        }
        while ready_tasks:
            scores = {}
            for task in ready_tasks:
                # get sub task graph with all previously scheduled tasks,
                # the current task and its k_depth successors
                k_depth_successors = nx.single_source_shortest_path_length(
                    G=task_graph,
                    source=task,
                    cutoff=self.k_depth
                )
                sub_task_graph = task_graph.subgraph(
                    set(scheduled_tasks.keys()) | set(k_depth_successors.keys()) | {task}
                )
                sub_schedule = self.scheduler.schedule(network, sub_task_graph, deepcopy(schedule))
                sub_schedule_makespan = max(
                    task.end for tasks in sub_schedule.values() for task in tasks
                )
                scores[task] = sub_schedule_makespan
            
            best_task = min(scores, key=scores.get)
            new_task = self.scheduler.insert_task(network, task_graph, schedule, best_task)
            scheduled_tasks[new_task.name] = new_task

            ready_tasks = {
                task for task in task_graph.nodes
                if (
                    all(pred in scheduled_tasks for pred in task_graph.predecessors(task))
                    and task not in scheduled_tasks
                )
            }

        return schedule
