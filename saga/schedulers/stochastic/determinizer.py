from typing import Callable, Dict, Hashable, List, Union

import networkx as nx

from saga.utils.random_variable import RandomVariable
from saga.scheduler import Scheduler, Task
from saga.utils.random_variable import RandomVariable

class Determinizer(Scheduler):
    def __init__(self,
                 scheduler: Scheduler,
                 determinize: Callable[[RandomVariable], float]) -> None:
        super().__init__()
        self.scheduler = scheduler
        self._determinize = determinize

    def determinize(self, rv: Union[RandomVariable, int, float]) -> float:
        if isinstance(rv, RandomVariable):
            return self._determinize(rv)
        return float(rv)

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: A dictionary mapping nodes to a list of tasks executed on the node.
        """
        det_network = nx.Graph()
        det_network.add_nodes_from(network.nodes)
        det_network.add_edges_from(network.edges)

        for node in det_network.nodes:
            det_network.nodes[node]["weight"] = self.determinize(network.nodes[node]["weight"])
            if det_network.nodes[node]["weight"] != det_network.nodes[node]["weight"]:
                raise ValueError("Node weight is nan")
        for edge in det_network.edges:
            det_network.edges[edge]["weight"] = self.determinize(network.edges[edge]["weight"])
            if det_network.edges[edge]["weight"] != det_network.edges[edge]["weight"]:
                raise ValueError("Edge weight is nan")

        det_task_graph = nx.DiGraph()
        det_task_graph.add_nodes_from(task_graph.nodes)
        det_task_graph.add_edges_from(task_graph.edges)
        for task in det_task_graph.nodes:
            # print(task, task_graph.nodes[task]["weight"], task_graph.out_degree(task), task_graph.in_degree(task))
            # if isinstance(task_graph.nodes[task]["weight"], RandomVariable):
                # print("samples:", task_graph.nodes[task]["weight"].samples)
            det_task_graph.nodes[task]["weight"] = self.determinize(task_graph.nodes[task]["weight"])
            if det_task_graph.nodes[task]["weight"] != det_task_graph.nodes[task]["weight"]:
                raise ValueError("Task weight is nan")
        for edge in det_task_graph.edges:
            det_task_graph.edges[edge]["weight"] = self.determinize(task_graph.edges[edge]["weight"])
            if det_task_graph.edges[edge]["weight"] != det_task_graph.edges[edge]["weight"]:
                raise ValueError("Edge weight is nan")


        return self.scheduler.schedule(det_network, det_task_graph)
