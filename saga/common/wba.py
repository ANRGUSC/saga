from typing import Dict, Hashable, List, Optional, Tuple
import networkx as nx
import numpy as np
import random

from ..base import Scheduler, Task
from ..utils.tools import check_instance_simple

class WBAScheduler(Scheduler):
    def __init__(self, alpha: float = 0.5) -> None:
        super(WBAScheduler, self).__init__()
        self.alpha = alpha

    def get_runtimes(self, network: nx.Graph, task_graph: nx.DiGraph) -> Tuple[Dict[Hashable, Dict[Hashable, float]], Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]]:
        runtimes = {}
        for node in network.nodes:
            runtimes[node] = {}
            speed: float = network.nodes[node]["weight"]
            for task in task_graph.nodes:
                cost: float = task_graph.nodes[task]["weight"]
                runtimes[node][task] = cost / speed

        commtimes = {}
        for src, dst in network.edges:
            commtimes[src, dst] = {}
            commtimes[dst, src] = {}
            speed: float = network.edges[src, dst]["weight"]
            for src_task, dst_task in task_graph.edges:
                cost = task_graph.edges[src_task, dst_task]["weight"]
                commtimes[src, dst][src_task, dst_task] = cost / speed
                commtimes[dst, src][src_task, dst_task] = cost / speed

        return runtimes, commtimes

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        check_instance_simple(network, task_graph)
        runtimes, commtimes = self.get_runtimes(network, task_graph)
        num_machines = len(network.nodes)
        num_tasks = len(task_graph.nodes)
        schedule = {node: [] for node in network.nodes}
        machines = {node: {'EAT': 0} for node in network.nodes}

        EET = np.zeros((num_tasks, num_machines))
        FAT = np.zeros((num_tasks, num_machines))

        tasks_not_mapped = list(task_graph.nodes)

        for i, task in enumerate(task_graph.nodes):
            for j, node in enumerate(network.nodes):
                EET[i, j] = self.runtimes[node][task]
                FAT[i, j] = max([self.commtimes[(node, schedule[dep_node][-1].node)][(dep_task[-1].name, task)] + dep_task[-1].end 
                                 for dep_node, dep_tasks in schedule.items() if dep_tasks and task_graph.has_edge(dep_tasks[-1].name, task)], 
                                 default=0)

        while tasks_not_mapped:
            avail_pairs = []
            I_min = float('inf')
            I_max = -float('inf')
            for task in tasks_not_mapped:
                task_index = list(task_graph.nodes).index(task)
                for machine_index, machine in enumerate(network.nodes):
                    I = max(EET[task_index, machine_index] + FAT[task_index, machine_index], machines[machine]['EAT']) - machines[machine]['EAT']
                    if I <= I_min + self.alpha * (I_max - I_min):
                        avail_pairs.append((task, machine))
                    I_min = min(I_min, I)
                    I_max = max(I_max, I)

            task, min_machine = random.choice(avail_pairs)
            task_obj = Task(min_machine, task, machines[min_machine]['EAT'], machines[min_machine]['EAT'] + EET[task_index, machine_index])
            schedule[min_machine].append(task_obj)
            machines[min_machine]['EAT'] = task_obj.end
            tasks_not_mapped.remove(task)

            for next_task in task_graph.successors(task):
                next_task_index = list(task_graph.nodes).index(next_task)
                for node in network.nodes:
                    if (task, next_task) in self.commtimes and (min_machine, node) in self.commtimes[(task, next_task)]:
                        FAT[next_task_index, list(network.nodes).index(node)] = max(FAT[next_task_index, list(network.nodes).index(node)], 
                                                   self.commtimes[(task, next_task)][(min_machine, node)] + machines[min_machine]['EAT'] if task != next_task else 0)

        return schedule
