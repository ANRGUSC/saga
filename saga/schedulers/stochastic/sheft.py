from typing import Dict, Hashable, List, Tuple, Union

import networkx as nx
from scipy.stats import rv_continuous

from saga.utils.random_variable import RandomVariable

from ...scheduler import Scheduler, Task
from ..heft import HeftScheduler


class SheftScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def reweight_instance(network: nx.Graph, task_graph: nx.DiGraph) -> Tuple[nx.Graph, nx.DiGraph]:
        """Re-weight the instance based on the expected value + 1 standard deviation.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Tuple[nx.Graph, nx.DiGraph]: The re-weighted instance.
        """
        # node/edge weights are scipy stats distributions
        # re-weight edges based on expected value + 1 standard deviation
        new_network = nx.Graph()
        for node in network.nodes:
            dist: Union[RandomVariable, rv_continuous] = network.nodes[node]["weight"]
            new_network.add_node(node, weight=dist.mean() + dist.std())
        for src, dst in network.edges:
            dist: Union[RandomVariable, rv_continuous] = network.edges[src, dst]["weight"]
            new_network.add_edge(src, dst, weight=dist.mean() + dist.std())

        new_task_graph = nx.DiGraph()
        for task in task_graph.nodes:
            dist: Union[RandomVariable, rv_continuous] = task_graph.nodes[task]["weight"]
            new_task_graph.add_node(task, weight=dist.mean() + dist.std())
        for src, dst in task_graph.edges:
            dist: Union[RandomVariable, rv_continuous] = task_graph.edges[src, dst]["weight"]
            new_task_graph.add_edge(src, dst, weight=dist.mean() + dist.std())

        return new_network, new_task_graph

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
        # node/edge weights are scipy stats distributions
        # re-weight edges based on expected value + 1 standard deviation
        new_network, new_task_graph = self.reweight_instance(network, task_graph)
        heft_scheduler = HeftScheduler()
        return heft_scheduler.schedule(new_network, new_task_graph)