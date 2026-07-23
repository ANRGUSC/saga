
import logging
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np

from saga.scheduler import Scheduler, Task


class HybridScheduler(Scheduler):
    """A hybrid scheduler."""
    def __init__(self, schedulers: Iterable[Scheduler]) -> None:
        """Initializes the hybrid scheduler.

        Args:
            schedulers (Iterable[Scheduler]): An iterable of schedulers.
        """
        self.schedulers = schedulers

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[str, List[Task]]:
        """Returns the best schedule of the given schedule functions.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[str, List[Task]]: The best schedule.
        """
        pass
        # best_scheduler, best_schedule, best_makespan = None, None, np.inf
        # for scheduler in self.schedulers:
        #     schedule = scheduler.schedule(network, task_graph)
        #     makespan = max(tasks[-1].end if tasks else 0 for tasks in schedule.values())
        #     if makespan < best_makespan:
        #         best_scheduler, best_schedule, best_makespan = scheduler, schedule, makespan
        # logging.debug("Best Scheduler: %s", best_scheduler.__class__.__name__)
        # return best_schedule
