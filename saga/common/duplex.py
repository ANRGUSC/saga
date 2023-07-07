from ..base import Scheduler, Task
from ..utils.tools import check_instance_simple

import itertools
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np

from .minmin import MinMinScheduler
from .maxmin import MaxMinScheduler

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
        minmin_schedule = MinMinScheduler().schedule(network, task_graph)
        maxmin_schedule = MaxMinScheduler().schedule(network, task_graph)

        minmin_makespan = max([
            max([task.end for task in minmin_schedule[node]])
            for node in minmin_schedule
        ])

        maxmin_makespan = max([
            max([task.end for task in maxmin_schedule[node]])
            for node in maxmin_schedule
        ])

        if minmin_makespan <= maxmin_makespan:
            return minmin_schedule
        else:
            return maxmin_schedule

        
    




        
        