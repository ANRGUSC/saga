from typing import Dict, Hashable, List, Tuple
import networkx as nx
from ....scheduler import Scheduler, Task, DWScheduler
import random


class NPRandomScheduler(DWScheduler):
    """Schedules all tasks on the node with the highest processing speed"""

    def schedule(self, 
                 network: nx.Graph, 
                 task_graphs: List[Tuple[nx.DiGraph, float]]
                 ) -> Dict[Hashable, List[Task]]:

        schedule = {node: [] for node in network.nodes}
        scheduled_tasks: Dict[Hashable, Task] = {}

 
        for i in range(len(task_graphs)):
            task_graph = task_graphs[i][0]
            task_graph_arrival_time = task_graphs[i][1]

            for task_name in nx.topological_sort(task_graph):
                # Choose one of the network nodes at random uniformly
                selected_node = random.choice(list(network.nodes))

                # calculate the time needed for dependencies to arrive
                if task_graph.in_degree(task_name) > 0:
                    data_arrival_time = max(
                        scheduled_tasks[pred].end + 
                        (task_graph.edges[pred, task_name]["weight"] / network.edges[selected_node, selected_node]["weight"])
                        for pred in task_graph.predecessors(task_name)
                    )
                else:
                    data_arrival_time = task_graph_arrival_time

                # calculate the time needed for the task to execute
                task_size = task_graph.nodes[task_name]["weight"]
                exec_time = task_size / network.nodes[selected_node]["weight"]
                start_time = max(data_arrival_time, task_graph_arrival_time, 
                                 schedule[selected_node][-1].end if schedule[selected_node] else 0)
                # create the task and add it to the schedule
                new_task = Task(selected_node, task_name, start_time, start_time + exec_time)
                schedule[selected_node].append(new_task)
                scheduled_tasks[task_name] = new_task

                # print the schedule

        return schedule
