from typing import Dict, Hashable, List, Tuple
import networkx as nx
from ....scheduler import Scheduler, Task, DWScheduler
import random


class PRandomScheduler(DWScheduler):
    """Schedules all tasks on the node with the highest processing speed"""

    def schedule(self, 
                 network: nx.Graph, 
                 task_graphs: List[Tuple[nx.DiGraph, float]]
                 ) -> Dict[Hashable, List[Task]]:
        

        for task_graph_tupple in task_graphs:
            for node in task_graph_tupple[0].nodes:
                task_graph_tupple[0].nodes[node]["arrival_time"] = task_graph_tupple[1]

        schedule = {node: [] for node in network.nodes}
        scheduled_tasks: Dict[Hashable, Task] = {}

 
        for i in range(len(task_graphs)):
            task_graph = nx.compose_all([task_graphs[j][0] for j in range(i + 1)])

            if i > 0:
                for task_name in task_graph.nodes:
                    matching_task = next((task for tasks in schedule.values() for task in tasks if task.name == task_name), None)
                    if matching_task:
                        if matching_task.start > task_graphs[i][1]:
                            schedule[matching_task.node].remove(matching_task)
                            scheduled_tasks.pop(task_name, None)


            for task_name in nx.topological_sort(task_graph):
                if task_name in scheduled_tasks:
                    continue
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
                    data_arrival_time = task_graph.nodes[task_name]["arrival_time"]

                # calculate the time needed for the task to execute
                task_size = task_graph.nodes[task_name]["weight"]
                exec_time = task_size / network.nodes[selected_node]["weight"]
                start_time = max(data_arrival_time, task_graph.nodes[task_name]["arrival_time"], 
                                 schedule[selected_node][-1].end if schedule[selected_node] else 0)
                # create the task and add it to the schedule
                new_task = Task(selected_node, task_name, start_time, start_time + exec_time)
                schedule[selected_node].append(new_task)
                scheduled_tasks[task_name] = new_task

                # print the schedule

        return schedule
