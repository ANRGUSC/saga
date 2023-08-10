from itertools import combinations, product
from pprint import pprint
from typing import Dict, Hashable, List
import networkx as nx

from saga.base import Task

from ..base import Scheduler, Task
from itertools import product
from typing import Dict, Hashable, List, Optional

import networkx as nx
from pulp import *

class PulpScheduler(Scheduler):
    def __init__(self) -> None:
        pass 

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        # create the LP object, set up as a minimization problem --> since we want to minimize the makespan
        prob = LpProblem("TaskScheduling", LpMinimize)

        # assuming network and task_graph are defined and have the same structure as in your original code

        # create decision variables
        start_time = {}
        is_assigned = {}
        for node in network.nodes:
            for task in task_graph.nodes:
                start_time[node, task] = LpVariable(f'start_time_{node}_{task}', lowBound=0, cat='Continuous')
                is_assigned[node, task] = LpVariable(f'is_assigned_{node}_{task}', cat='Binary')

        makespan = LpVariable("makespan", lowBound=0, cat='Continuous')

        # objective function: minimize makespan
        prob += makespan

        # constraint: is_assigned[node, task] == 0 => start_time[node, task] < 0
        for node in network.nodes:
            for task in task_graph.nodes:
                prob += is_assigned[node, task] * -1 <= start_time[node, task]

        # constraint: each task is assigned to exactly one node
        for task in task_graph.nodes:
            prob += lpSum(is_assigned[node, task] for node in network.nodes) == 1

        # constraint: each node can only execute one task at a time
        for node in network.nodes:
            for task1, task2 in combinations(task_graph.nodes, r=2):
                # ensure task1 ends before task2 starts or task2 ends before task1 starts
                prob += start_time[node, task1] + task_graph.nodes[task1]['weight'] / network.nodes[node]['weight'] <= start_time[node, task2]
                prob += start_time[node, task2] + task_graph.nodes[task2]['weight'] / network.nodes[node]['weight'] <= start_time[node, task1]

        # communication time constraints
        for task1, task2 in task_graph.edges:
            for node1, node2 in product(network.nodes, repeat=2):
                # task2 start time is at least task1 end time + communication time
                prob += start_time[node1, task1] + task_graph.nodes[task1]['weight'] / network.nodes[node1]['weight'] + task_graph.edges[task1, task2]['weight'] / network.edges[node1, node2]['weight'] <= start_time[node2, task2]

        # makespan constraint
        for node in network.nodes:
            for task in task_graph.nodes:
                # end time of each task should be less than or equal to makespan
                prob += start_time[node, task] + task_graph.nodes[task]['weight'] / network.nodes[node]['weight'] <= makespan

        # solve the problem
        status = prob.solve()
        print('Status:', LpStatus[status])

        if LpStatus[status] == 'Optimal':
            schedule = {}
            # create schedule
            for node in network.nodes:
                schedule[node] = []
                for task in task_graph.nodes:
                    schedule[node].append(
                        Task(
                            node=node,
                            name=task,
                            start=value(start_time[node, task]),
                            end=value(start_time[node, task]) + task_graph.nodes[task]['weight'] / network.nodes[node]['weight']
                        )
                    )
            # sort tasks by start time
            for node in schedule:
                schedule[node].sort(key=lambda task: task.start)
            print('Schedule:', schedule)
            print('Makespan:', value(makespan))
            return schedule
        else:
            print('No optimal schedule found')
            return {}

    
def test():
    task_graph = nx.DiGraph()
    task_graph.add_node('A', weight=1)
    task_graph.add_node('B', weight=2)
    task_graph.add_node('C', weight=1)
    task_graph.add_node('D', weight=3)

    task_graph.add_edge('A', 'B', weight=1)
    task_graph.add_edge('A', 'C', weight=2)
    task_graph.add_edge('B', 'D', weight=3)
    task_graph.add_edge('C', 'D', weight=1)

    network = nx.Graph()
    network.add_node(1, weight=1)
    network.add_node(2, weight=2)
    network.add_node(3, weight=3)
    network.add_node(4, weight=4)

    for node1, node2 in product(network.nodes, network.nodes):
        network.add_edge(node1, node2, weight=1e9 if node1 == node2 else 1)

    scheduler = PulpScheduler()
    schedule = scheduler.schedule(network, task_graph)
    pprint(schedule)

if __name__ == "__main__":
    test()
