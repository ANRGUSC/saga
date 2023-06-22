from ..base import Scheduler, Task
from ..utils.tools import check_instance_simple

import itertools
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np

class DuplexScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Returns the best schedule (minimizing makespan) for a problem instance using dupkex

        Args:
            network: Network
            task_graph: Task graph
        
        Returns:
            A dictionary of the schedule
        """
        check_instance_simple(network, task_graph)
        
        # get all topological sorts of the task graph
        topological_sorts = list(nx.algorithms.dag.all_topological_sorts(task_graph))
        
        