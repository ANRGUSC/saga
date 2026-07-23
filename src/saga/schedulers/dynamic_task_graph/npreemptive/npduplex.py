from typing import Dict, Hashable, List, Tuple

import networkx as nx

from ....scheduler import Scheduler, Task
from .npmaxmin import NPMaxMinScheduler
from .npminmin import NPMinMinScheduler


class ResidualDuplexScheduler(Scheduler): # pylint: disable=too-few-public-methods
    """Duplex scheduler"""
    def schedule(self, network: nx.Graph, task_graphs: List[Tuple[nx.DiGraph, float]]) -> Dict[Hashable, List[Task]]:
        """Returns the best schedule (minimizing makespan) for a problem instance using dupkex

        Args:
            network: Network
            task_graph: Task graph
        
        Returns:
            A dictionary of the schedule
        """
        minmin_schedule = NPMinMinScheduler().schedule(network, task_graphs)
        maxmin_schedule = NPMaxMinScheduler().schedule(network, task_graphs)


        minmin_makespan = max([0 if not tasks else tasks[-1].end for tasks in minmin_schedule.values()])

        maxmin_makespan = max([0 if not tasks else tasks[-1].end for tasks in maxmin_schedule.values()])


        if minmin_makespan <= maxmin_makespan:
            return minmin_schedule
        return maxmin_schedule
