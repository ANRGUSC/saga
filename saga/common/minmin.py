# from ..base import Scheduler, Task
# from ..utils.tools import check_instance_simple

# from typing import Dict, Hashable, List

# import networkx as nx

# class MinMinScheduler(Scheduler):
#     def __init__(self):
#         super().__init__()

#     def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
#         """Returns the best schedule (minimizing makespan) for a problem instance using the MinMin Algorithm.

#         Args:
#             network: Network
#             task_graph: Task graph

#         Returns:
#             A dictionary of the schedule
#         """
#         check_instance_simple(network, task_graph)

#         # Initialize schedule as an empty dictionary
#         schedule = {node: [] for node in network.nodes}

#         # While there are unscheduled tasks
#         while task_graph.number_of_nodes() > 0:
#             # Calculate earliest completion time for each task on each machine
#             ect = {}
#             for task in task_graph.nodes:
#                 for machine in network.nodes:
#                     communication_time = max([task_graph.edges[pred, task]['weight'] for pred in task_graph.predecessors(task)] or [0])
#                     execution_time = task_graph.nodes[task]['weight'] / network.nodes[machine]['weight']
#                     completion_time = max([task.end for task in schedule[machine]] or [0]) + communication_time + execution_time
#                     ect[(task, machine)] = completion_time

#             # Find task-machine pair with minimum earliest completion time
#             min_task, min_machine = min(ect, key=ect.get)

#             # Add the task to the schedule
#             task = Task(start=max([task.end for task in schedule[min_machine]] or [0]),
#                         end=ect[(min_task, min_machine)],
#                         task=min_task,
#                         machine=min_machine)
#             schedule[min_machine].append(task)

#             # Remove the scheduled task from the task graph
#             task_graph.remove_node(min_task)

#         return schedule

      
# Sanjana implementation
from itertools import product
from typing import Dict, Hashable, List, Optional
import networkx as nx
import numpy as np

from ..base import Scheduler, Task
from ..utils.tools import check_instance_simple

class MinMinScheduler(Scheduler):
    def __init__(self):
        super(MinMinScheduler, self).__init__()

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
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
            print(f"  EAT for node {node}: {eat}")
            return eat
        
        # FAT = np.zeros((num_tasks, num_machines))
        def FAT(task: Hashable, node: Hashable) -> float:
            fat = 0 if task_graph.in_degree(task) <= 0 else max([
                # schedule[r_schedule[pred_task]][-1].end + 
                scheduled_tasks[pred_task].end +
                COMMTIME(pred_task, task, scheduled_tasks[pred_task].node, node)
                for pred_task in task_graph.predecessors(task)
            ]) 
            print(f"  FAT for task {task} on node {node}: {fat}")
            return fat
        
        def ECT(task: Hashable, node: Hashable) -> float:
            return EET(task, node) + max(EAT(node), FAT(task, node))
        
        while len(scheduled_tasks) < task_graph.order():
            available_tasks = [
                task for task in task_graph.nodes
                if task not in scheduled_tasks and set(task_graph.predecessors(task)).issubset(set(scheduled_tasks.keys()))
            ]
            while available_tasks:
                sched_task, sched_node = min(
                    product(available_tasks, network.nodes),
                    key=lambda instance: ECT(instance[0], instance[1])
                )
                print(f"ECTS")
                for (task, node) in product(available_tasks, network.nodes):
                    print(f"  Task {task} on node {node}: {ECT(task, node)}")
                schedule.setdefault(sched_node, [])
                new_task = Task(
                    node=sched_node,
                    name=sched_task,
                    start=EAT(sched_node),
                    end=ECT(sched_task, sched_node)
                )
                schedule[sched_node].append(new_task)
                scheduled_tasks[sched_task] = new_task
                print(f"Task {sched_task} scheduled on machine {sched_node}")
                available_tasks.remove(sched_task)

        return schedule


        # for i, task in enumerate(task_graph.nodes):
        #     for j, node in enumerate(network.nodes):
        #         EET[i, j] = task_graph.nodes[task]['weight'] / network.nodes[node]['weight']
        #         FAT[i, j] = max([self.commtimes[(task, dep_task)][(node, schedule[dep_node][-1].end)] + machines[dep_node]['EAT'] 
        #                          for dep_node, dep_task in schedule.items() if dep_task and task_graph.has_edge(dep_task[-1].name, task)], 
        #                          default=0)

        # for task in task_graph.nodes:
        #     # print(f"Processing task {task}")
        #     # print(f"Task resources required: {self.task_resources[task]}")
        #     # for machine in machines:
        #     #     print(f"Machine {machine} current resources: {machines[machine]['resources']}")
        #     #     print(f"Machine {machine} capacity: {self.machine_capacities[machine]}")
        #     # viable_machines = [machine for machine in machines if self.machine_capacities[machine] - machines[machine]['resources'] >= self.task_resources[task]]
        #     # if not viable_machines:
        #     #     print(f"No viable machines for task {task}")
        #     #     raise ValueError(f"No viable machines for task {task}")
        #     task_index = list(task_graph.nodes).index(task)
        #     machine_indices = [list(network.nodes).index(machine) for machine in machines]
        #     min_machine_index = machine_indices[np.argmin(EET[task_index, machine_indices] + np.maximum(FAT[task_index, machine_indices], [machines[machine]['EAT'] for machine in machines]))]
        #     min_machine = list(network.nodes)[min_machine_index]

        #     # start_time = 0
        #     # end_time = 1

        #     task_obj = Task(min_machine, task, start_time, end_time)
        #     task_obj.node = min_machine
        #     task_obj.name = task
        #     task_obj.start = machines[min_machine]['EAT']
        #     task_obj.end = machines[min_machine]['EAT'] + EET[task_index, min_machine_index]

        #     schedule[min_machine].append(task_obj)

        #     machines[min_machine]['EAT'] += EET[task_index, min_machine_index]
        #     machines[min_machine]['resources'] += self.task_resources[task]

        #     for next_task in task_graph.successors(task):
        #         next_task_index = list(task_graph.nodes).index(next_task)
        #         for node in network.nodes:
        #             if ((task, next_task) in self.commtimes) and ((min_machine, node) in self.commtimes[(task, next_task)]):
        #                 FAT[next_task_index, list(network.nodes).index(node)] = max(FAT[next_task_index, list(network.nodes).index(node)], 
        #                                            self.commtimes[(task, next_task)][(min_machine, node)] + machines[min_machine]['EAT'] if task != next_task else 0)

        return schedule