from typing import Dict, Hashable, List, Tuple
import networkx as nx
from ....scheduler import Scheduler, Task, DWScheduler


class PFastestNodeScheduler(DWScheduler):
    """Schedules all tasks on the node with the highest processing speed"""

    def schedule(self, 
                 network: nx.Graph, 
                 task_graphs: List[Tuple[nx.DiGraph, float]]
                 ) -> Dict[Hashable, List[Task]]:
        """Schedules all tasks on the node with the highest processing speed

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: A schedule mapping nodes to a list of tasks.

        Raises:
            ValueError: If the instance is not valid
        """
        fastest_node = max(network.nodes, key=lambda node: network.nodes[node]["weight"])
        schedule = {node: [] for node in network.nodes}
        scheduled_tasks: Dict[Hashable, Task] = {}
        # add tasks to fastest node in order (topological sort)
        free_time = 0
 
        for i in range(len(task_graphs)):
            task_graph = nx.compose_all([task_graphs[j][0] for j in range(i + 1)])
            task_graph_arrival_time = task_graphs[i][1]

            if i > 0:
                for task_name in task_graph.nodes:
                    matching_task = next((task for tasks in schedule.values() for task in tasks if task.name == task_name), None)
                    if matching_task:
                        if matching_task.start > task_graphs[i][1]:
                            schedule[matching_task.node].remove(matching_task)
                            scheduled_tasks.pop(task_name, None)

            

            for task_name in nx.topological_sort(task_graph):
                task_size = task_graph.nodes[task_name]["weight"]
                exec_time = task_size / network.nodes[fastest_node]["weight"]

                # For most instances, the data should probably arrive immediately
                # since everything is executing on the same node.
                data_arrival_time = task_graph_arrival_time
                if task_graph.in_degree(task_name) > 0:
                    data_arrival_time = max(
                        scheduled_tasks[pred].end + (
                            task_graph.edges[pred, task_name]["weight"] / network.edges[fastest_node, fastest_node]["weight"]
                        )
                        for pred in task_graph.predecessors(task_name)
                    )
                start_time = max(free_time, data_arrival_time)

                new_task = Task(fastest_node, task_name, start_time, start_time + exec_time)
                schedule[fastest_node].append(new_task)
                scheduled_tasks[task_name] = new_task
                free_time = new_task.end

        return schedule
