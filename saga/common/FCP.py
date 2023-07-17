from ..base import Scheduler, Task
from ..utils.tools import check_instance_simple
from itertools import product

import itertools
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np
class FCPScheduler(Scheduler):
    def __init__(self):
        super(FCPScheduler, self).__init__()

    def CriticalPath(self, task_graph: nx.DiGraph) -> List[Hashable]:
        """Returns the critical path of a task graph

        Args:
            task_graph: Task graph

        Returns:
            A list of tasks in the critical path
        """
        distance = {node: (0, []) for node in task_graph.nodes}
    
        for node in nx.topological_sort(task_graph):
            # We're only interested in the longest path
            score, path = max(
                [(distance[node][0], path + [node]) if node in path else (0, []) for path in distance[node][1]],
                key = lambda x: x[0]
            )
            for successor in task_graph.successors(node):
                if distance[successor][0] < score + task_graph.edges[node, successor]['weight']:
                    distance[successor] = (score + task_graph.edges[node, successor]['weight'], path)
        node, (length, path) = max(distance.items(), key=lambda x: x[1][0])
        
        return path

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Returns the best schedule (minimizing makespan) for a problem instance using FCP(Fastest Critical Path)

        Args:
            network: Network
            task_graph: Task graph

        Returns:
            A dictionary of the schedule
        """
        check_instance_simple(network, task_graph)

        schedule: Dict[Hashable, List[Task]] = {}
        scheduled_tasks: Dict[Hashable, Task] = {} # Map from task_name to Task

        def EET(task: Hashable, node: Hashable) -> float:
            return task_graph.nodes[task]['weight'] / network.nodes[node]['weight']
        
        def COMMTIME(task1: Hashable, task2: Hashable, node1: Hashable, node2: Hashable) -> float:
            return task_graph.edges[task1, task2]['weight'] / network.edges[node1, node2]['weight']

        # EAT = {node: 0 for node in network.nodes}
        def EAT(node: Hashable) -> float:
            eat = schedule[node][-1].end if schedule.get(node) else 0
            return eat
        
        # FAT = np.zeros((num_tasks, num_machines))
        def FAT(task: Hashable, node: Hashable) -> float:
            fat = 0 if task_graph.in_degree(task) <= 0 else max([
                # schedule[r_schedule[pred_task]][-1].end + 
                scheduled_tasks[pred_task].end +
                COMMTIME(pred_task, task, scheduled_tasks[pred_task].node, node)
                for pred_task in task_graph.predecessors(task)
            ]) 
            return fat
        
        def ECT(task: Hashable, node: Hashable) -> float:
            return EET(task, node) + max(EAT(node), FAT(task, node))
        
        # Get the critical path
        critical_path = self.CriticalPath(task_graph)

        all_tasks = list(task_graph.nodes)
        ready_tasks = [task for task in all_tasks if task_graph.in_degree(task) == 0]

        while ready_tasks:
            critical_ready_tasks = [task for task in ready_tasks if task in critical_path]
            non_critical_ready_tasks = [task for task in ready_tasks if task not in critical_path]

            if critical_ready_tasks:
                sched_task, sched_node = min(
                    product(critical_ready_tasks, network.nodes),
                    key=lambda instance: ECT(instance[0], instance[1])
                )
            else:
                sched_task, sched_node = min(
                    product(non_critical_ready_tasks, network.nodes),
                    key=lambda instance: ECT(instance[0], instance[1])
                )

            schedule.setdefault(sched_node, [])
            new_task = Task(
                node=sched_node,
                name=sched_task,
                start=max(EAT(sched_node), FAT(sched_task, sched_node)),
                end=ECT(sched_task, sched_node)
            )
            schedule[sched_node].append(new_task)
            scheduled_tasks[sched_task] = new_task
            ready_tasks.remove(sched_task)

            for succ_task in task_graph.successors(sched_task):
                if all(pred_task in scheduled_tasks for pred_task in task_graph.predecessors(succ_task)):
                    ready_tasks.append(succ_task)

        return schedule
