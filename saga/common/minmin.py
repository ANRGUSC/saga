from ..base import Scheduler, Task
from ..utils.tools import check_instance_simple

import itertools
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np


class MinMinScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Returns the best schedule (minimizing makespan) for a problem instance using Minmin

        Args:
            network: Network
            task_graph: Task graph

        Returns:
            A dictionary of the schedule
        """
        check_instance_simple(network, task_graph)

        # Initialize schedule as an empty dictionary
        schedule = {node: [] for node in network.nodes}

        # While there are unscheduled tasks
        while task_graph.number_of_nodes() > 0:
            # Calculate earliest completion time for each task on each machine
            ect = {}
            for task in task_graph.nodes:
                for machine in network.nodes:
                    communication_time = max(
                        [task_graph.edges[pred, task]['weight'] for pred in task_graph.predecessors(task)] or [0])
                    execution_time = task_graph.nodes[task]['weight'] / \
                        network.nodes[machine]['weight']
                    completion_time = max([task.end for task in schedule[machine]] or [
                                          0]) + communication_time + execution_time
                    ect[(task, machine)] = completion_time

            # Find task-machine pair with minimum earliest completion time
            min_task, min_machine = min(ect, key=ect.get)

            # Add the task to the schedule
            task = Task(start=max([task.end for task in schedule[min_machine]] or [0]),
                        end=ect[(min_task, min_machine)],
                        task=min_task,
                        machine=min_machine)
            schedule[min_machine].append(task)

            # Remove the scheduled task from the task graph
            task_graph.remove_node(min_task)

        return schedule

      
# Sanjana implementation
# from typing import Dict, Hashable, List, Optional
# import networkx as nx
# import numpy as np

# from ..base import Scheduler, Task
# from ..utils.tools import check_instance_simple
# 
# class MinMinScheduler(Scheduler):
#     def __init__(self, runtimes, commtimes, machine_capacities, task_resources):
#         super(MinMinScheduler, self).__init__()
#         self.runtimes = runtimes
#         self.commtimes = commtimes
#         self.machine_capacities = machine_capacities
#         self.task_resources = task_resources
# 
#     def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
#         check_instance_simple(network, task_graph)
#         num_machines = len(network.nodes)
#         num_tasks = len(task_graph.nodes)
#         schedule = {node: [] for node in network.nodes}
#         machines = {node: {'EAT': 0, 'resources': 0} for node in network.nodes}
# 
#         EET = np.zeros((num_tasks, num_machines))
#         FAT = np.zeros((num_tasks, num_machines))
# 
#         for i, task in enumerate(task_graph.nodes):
#             for j, node in enumerate(network.nodes):
#                 EET[i, j] = self.runtimes[node][task]
#                 FAT[i, j] = max([self.commtimes[(task, dep_task)][(node, schedule[dep_node][-1].end)] + machines[dep_node]['EAT'] 
#                                  for dep_node, dep_task in schedule.items() if dep_task and task_graph.has_edge(dep_task[-1].name, task)], 
#                                  default=0)
# 
#         for task in task_graph.nodes:
#             # print(f"Processing task {task}")
#             # print(f"Task resources required: {self.task_resources[task]}")
#             # for machine in machines:
#             #     print(f"Machine {machine} current resources: {machines[machine]['resources']}")
#             #     print(f"Machine {machine} capacity: {self.machine_capacities[machine]}")
#             # viable_machines = [machine for machine in machines if self.machine_capacities[machine] - machines[machine]['resources'] >= self.task_resources[task]]
#             # if not viable_machines:
#             #     print(f"No viable machines for task {task}")
#             #     raise ValueError(f"No viable machines for task {task}")
#             task_index = list(task_graph.nodes).index(task)
#             machine_indices = [list(network.nodes).index(machine) for machine in machines]
#             min_machine_index = machine_indices[np.argmin(EET[task_index, machine_indices] + np.maximum(FAT[task_index, machine_indices], [machines[machine]['EAT'] for machine in machines]))]
#             min_machine = list(network.nodes)[min_machine_index]
# 
#             # start_time = 0
#             # end_time = 1
# 
#             task_obj = Task(min_machine, task, start_time, end_time)
#             task_obj.node = min_machine
#             task_obj.name = task
#             task_obj.start = machines[min_machine]['EAT']
#             task_obj.end = machines[min_machine]['EAT'] + EET[task_index, min_machine_index]
# 
#             schedule[min_machine].append(task_obj)
# 
#             machines[min_machine]['EAT'] += EET[task_index, min_machine_index]
#             machines[min_machine]['resources'] += self.task_resources[task]
# 
#             for next_task in task_graph.successors(task):
#                 next_task_index = list(task_graph.nodes).index(next_task)
#                 for node in network.nodes:
#                     if ((task, next_task) in self.commtimes) and ((min_machine, node) in self.commtimes[(task, next_task)]):
#                         FAT[next_task_index, list(network.nodes).index(node)] = max(FAT[next_task_index, list(network.nodes).index(node)], 
#                                                    self.commtimes[(task, next_task)][(min_machine, node)] + machines[min_machine]['EAT'] if task != next_task else 0)
# 
#         return schedule