from typing import Dict, Hashable, List

import networkx as nx

from ..scheduler import Scheduler, Task
from .maxmin import MaxMinScheduler
from .minmin import MinMinScheduler


class DuplexScheduler(Scheduler): # pylint: disable=too-few-public-methods
    """Duplex scheduler"""
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

        minmin_makespan = max(
            max(task.end for task in tasks)
            for _, tasks in minmin_schedule.items()
        )

        maxmin_makespan = max(
            max(task.end for task in tasks)
            for _, tasks in maxmin_schedule.items()
        )

        if minmin_makespan <= maxmin_makespan:
            return minmin_schedule
        return maxmin_schedule
