from typing import Dict, Hashable, List
import networkx as nx
from ..base import Scheduler, Task
from ..utils.tools import check_instance_simple

class FastestNodeScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__()

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
        check_instance_simple(network, task_graph)
        fastest_node = max(network.nodes, key=lambda node: network.nodes[node]["weight"])
        schedule = {node: [] for node in network.nodes}
        # add tasks to fastest node in order (topological sort)
        free_time = 0
        for task_name in nx.topological_sort(task_graph):
            task_size = task_graph.nodes[task_name]["weight"]
            exec_time = task_size / network.nodes[fastest_node]["weight"]
            schedule[fastest_node].append(Task(fastest_node, task_name, free_time, free_time + exec_time))
            free_time += exec_time
        return schedule