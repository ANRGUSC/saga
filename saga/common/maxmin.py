from typing import Dict, Hashable, List, Optional
import networkx as nx
import numpy as np

from ..base import Scheduler, Task
from ..utils.tools import check_instance_simple

# class MaxMinScheduler(Scheduler):
#     def __init__(self, runtimes, commtimes, machine_capacities, task_resources):
#         super(MaxMinScheduler, self).__init__()
#         self.runtimes = runtimes
#         self.commtimes = commtimes
#         self.machine_capacities = machine_capacities
#         self.task_resources = task_resources

#     def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
#         check_instance_simple(network, task_graph)
#         num_machines = len(network.nodes)
#         num_tasks = len(task_graph.nodes)
#         schedule = {node: [] for node in network.nodes}
#         machines = {node: {'EAT': 0, 'resources': 0} for node in network.nodes}

#         EET = np.zeros((num_tasks, num_machines))
#         FAT = np.zeros((num_tasks, num_machines))

#         for i, task in enumerate(task_graph.nodes):
#             for j, node in enumerate(network.nodes):
#                 EET[i, j] = self.runtimes[node][task]
#                 FAT[i, j] = max([self.commtimes[(task, dep_task)][(node, schedule[dep_node][-1].end)] + machines[dep_node]['EAT'] 
#                                  for dep_node, dep_task in schedule.items() if dep_task and task_graph.has_edge(dep_task[-1].name, task)], 
#                                  default=0)

#         # A list of all tasks for processing
#         all_tasks = list(task_graph.nodes)
#         while all_tasks:
#             # Get the task index and machine index of the maximum completion time among all tasks and viable machines
#             max_completion_task_index, max_completion_machine_index = max(
#                 ((task_index, machine_index) 
#                 for task_index, task in enumerate(all_tasks) 
#                 for machine_index, machine in enumerate(network.nodes) 
#                 if self.machine_capacities[machine] - machines[machine]['resources'] >= self.task_resources[task]), 
#                 key=lambda pair: EET[pair[0], pair[1]] + max(FAT[pair[0], pair[1]], machines[list(network.nodes)[pair[1]]]['EAT'])
#             )
#             max_completion_task = all_tasks[max_completion_task_index]
#             max_completion_machine = list(network.nodes)[max_completion_machine_index]

#             start_time = machines[max_completion_machine]['EAT']
#             end_time = machines[max_completion_machine]['EAT'] + EET[max_completion_task_index, max_completion_machine_index]

#             task_obj = Task(max_completion_machine, max_completion_task, start_time, end_time)
#             task_obj.node = max_completion_machine
#             task_obj.name = max_completion_task
#             task_obj.start = start_time
#             task_obj.end = end_time

#             schedule[max_completion_machine].append(task_obj)

#             machines[max_completion_machine]['EAT'] += EET[max_completion_task_index, max_completion_machine_index]
#             machines[max_completion_machine]['resources'] += self.task_resources[max_completion_task]

#             for next_task in task_graph.successors(max_completion_task):
#                 next_task_index = list(task_graph.nodes).index(next_task)
#                 for node in network.nodes:
#                     if ((max_completion_task, next_task) in self.commtimes) and ((max_completion_machine, node) in self.commtimes[(max_completion_task, next_task)]):
#                         FAT[next_task_index, list(network.nodes).index(node)] = max(FAT[next_task_index, list(network.nodes).index(node)], 
#                                                    self.commtimes[(max_completion_task, next_task)][(max_completion_machine, node)] + machines[max_completion_machine]['EAT'] if max_completion_task != next_task else 0)

#             # Remove the task with the maximum completion time from the list of all tasks
#             all_tasks.pop(max_completion_task_index)

#         return schedule

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
                    start=EAT(sched_node),
                    end=ECT(sched_task, sched_node)
                )
                schedule[sched_node].append(new_task)
                scheduled_tasks[sched_task] = new_task
                
                available_tasks.remove(sched_task)

        return schedule
