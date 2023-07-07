from typing import Dict, Hashable, List
import networkx as nx

from ..base import Scheduler, Task
from ..utils.tools import check_instance_simple

class MaxMinScheduler(Scheduler):
    def __init__(self):
        super(MaxMinScheduler, self).__init__()

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        check_instance_simple(network, task_graph)

        schedule: Dict[Hashable, List[Task]] = {}
        scheduled_tasks: Dict[Hashable, Task] = {} # Map from task_name to Task

        def EET(task: Hashable, node: Hashable) -> float:
            return task_graph.nodes[task]['weight'] / network.nodes[node]['weight']
        
        def COMMTIME(task1: Hashable, task2: Hashable, node1: Hashable, node2: Hashable) -> float:
            return task_graph.edges[task1, task2]['weight'] / network.edges[node1, node2]['weight']

        def EAT(node: Hashable) -> float:
            return schedule[node][-1].end if schedule.get(node) else 0
        
        def FAT(task: Hashable, node: Hashable) -> float:
            return 0 if task_graph.in_degree(task) <= 0 else max([
                scheduled_tasks[pred_task].end +
                COMMTIME(pred_task, task, scheduled_tasks[pred_task].node, node)
                for pred_task in task_graph.predecessors(task)
            ])
        
        def ECT(task: Hashable, node: Hashable) -> float:
            return EET(task, node) + max(EAT(node), FAT(task, node))
        
        while len(scheduled_tasks) < task_graph.order():
            available_tasks = [
                task for task in task_graph.nodes
                if task not in scheduled_tasks and all(pred in scheduled_tasks for pred in task_graph.predecessors(task))
            ]
            #slight change from MinMin logic
            while available_tasks:
                min_ects = {task: min(ECT(task, node) for node in network.nodes) for task in available_tasks}
                sched_task = max(min_ects, key=min_ects.get)
                sched_node = min(network.nodes, key=lambda node: ECT(sched_task, node))
                
                schedule.setdefault(sched_node, [])
                new_task = Task(
                    node=sched_node,
                    name=sched_task,
                    start=max(EAT(sched_node), FAT(sched_task, sched_node)),
                    end=ECT(sched_task, sched_node)
                )
                schedule[sched_node].append(new_task)
                scheduled_tasks[sched_task] = new_task
                
                available_tasks.remove(sched_task)

        return schedule
