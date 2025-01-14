from saga.schedulers import HeftScheduler
from saga.scheduler import Scheduler, Task
# from saga.utils.random_variable import RandomVariable

from typing import Dict, Hashable, List
import networkx as nx

class OnlineHeftScheduler(Scheduler):
    def __init__(self):
        self.heft_scheduler = HeftScheduler()

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        # Assume network and task graph node/edge attributes are:
        #   - weight_actual: float
        #   - weight_estimate: float

        schedule: Dict[Hashable, List[Task]] = {
            node: [] for node in network.nodes
        }
        while True:
            # Step 1: set weight attributes to weight_actual if the task is in the schedule
            #         otherwise, set to weight_estimate

            new_schedule = self.heft_scheduler.schedule(network, task_graph, schedule)

            # Step 2: Get next task to finish based on *actual* execution time
            # Step 3: Remove all tasks except those already finished or currently running
            # Step 4: Repeat until there are no more tasks to schedule


        
        pass
