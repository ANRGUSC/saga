import pathlib
from typing import List, Optional
import numpy as np

from saga import Schedule, Scheduler, ScheduledTask, TaskGraph, Network, TaskGraphNode, NetworkNode
from saga.schedulers.heft import heft_rank_sort
from saga.schedulers.cpop import upward_rank

thisdir = pathlib.Path(__file__).resolve().parent


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
        #maybe add a ready queue, and why is the task 4 being pushed back so far?
        self.network = network
        self.task_graph = task_graph
        schedule = Schedule(task_graph, network)
        #non-dominated sorting
        non_dominated_tasks: List[TaskGraphNode] = []
        dominated_tasks: List[TaskGraphNode] = [Task for Task in task_graph.tasks]
        #length = len(dominated_tasks)
        #initially put task 0 in the non-dominated list
        non_dominated_tasks.append(dominated_tasks.pop(0))
        
        for dom_task in list(dominated_tasks):
            to_remove = []
            is_dominated = False
            for ndom_task in non_dominated_tasks:
                if self.dominates(ndom_task, dom_task):
                    is_dominated = True
                    break
                elif self.dominates(dom_task, ndom_task):
                    # non_dominated_tasks.append(dom_task)
                    to_remove.append(ndom_task)
            if not is_dominated:
                for task in to_remove:
                    non_dominated_tasks.remove(task)
                    dominated_tasks.append(task)
                dominated_tasks.remove(dom_task)
                non_dominated_tasks.append(dom_task)

        #scheduling tasks
        network_nodes: List[NetworkNode] = sorted(
            network.nodes, key=lambda n: n.speed, reverse=True
        )
        #sorting tasks
        sorted_tasks = (
            sorted(non_dominated_tasks, key=lambda t: t.cost, reverse=True)
            + sorted(dominated_tasks, key=lambda t: t.cost, reverse=True)
        )

        j = 0
        for task in sorted_tasks:
            if j >= 0:
                start_time = schedule.get_earliest_start_time(task, network_nodes[j].name) #need to adapt
                newtask = ScheduledTask(
                    name=task.name,
                    node=network_nodes[j].name,
                    start=start_time, 
                    end=start_time + task.cost / network_nodes[j].speed,
                )
                schedule.add_task(newtask)
                j += 1
                if j >= len(network_nodes):
                    j = 0
        return schedule
        
        
        


    def dominates(self, task1: TaskGraphNode, task2: TaskGraphNode) -> bool:
        """Check if task1 dominates task2.

        Args:
            task1 (TaskGraphNode): The first task.
            task2 (TaskGraphNode): The second task. 
        Returns:
            bool: True if task1 dominates task2, False otherwise.   
        """
        #for this example, we will use size and upward rank, but in the future we should track wait time.
        rank_order = heft_rank_sort(network=self.network, task_graph=self.task_graph)
        rankings = {name: i for i, name in enumerate(rank_order)}
        return (
            (task1.cost < task2.cost and rankings[task1.name] < rankings[task2.name])
            or (task1.cost <= task2.cost and rankings[task1.name] < rankings[task2.name]) 
            or (task1.cost < task2.cost and rankings[task1.name] <= rankings[task2.name])
        )

        

        