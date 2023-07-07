from typing import Dict, List, Hashable 
import networkx as nx

from ..base import Scheduler, Task
from ..utils.tools import check_instance_simple

class METScheduler(Scheduler):
    def __init__(self):
        super(METScheduler, self).__init__()

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        check_instance_simple(network, task_graph)

        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}  # Initialize list for each node
        scheduled_tasks: Dict[Hashable, Task] = {} # Map from task_name to Task

        def ExecTime(task: Hashable, node: Hashable) -> float:
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
        
        for task in nx.topological_sort(task_graph):
            # Find node with minimum execution time for the task
            sched_node = min(network.nodes, key=lambda node: ExecTime(task, node))

            start_time = max(EAT(sched_node), FAT(task, sched_node))
            end_time = start_time + ExecTime(task, sched_node)

            # Add task to the schedule
            new_task = Task(node=sched_node, name=task, start=start_time, end=end_time)
            schedule[sched_node].append(new_task)
            scheduled_tasks[task] = new_task
            

        return schedule
