from typing import Dict, Hashable, List
import networkx as nx
from ..scheduler import Scheduler, Task


class FastestNodeScheduler(Scheduler):
    """Schedules all tasks on the node with the highest processing speed"""

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
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
        for task_name in nx.topological_sort(task_graph):
            task_size = task_graph.nodes[task_name]["weight"]
            exec_time = task_size / network.nodes[fastest_node]["weight"]

            # For most instances, the data should probably arrive immediately
            # since everything is executing on the same node.
            data_arrival_time = 0
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
