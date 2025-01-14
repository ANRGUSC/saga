from saga.schedulers import HeftScheduler
from saga.scheduler import Scheduler, Task
# from saga.utils.random_variable import RandomVariable

from typing import Dict, Hashable, List
import networkx as nx

def schedule_estimate_to_actual(network: nx.Graph,
                                task_graph: nx.DiGraph,
                                schedule_estimate: Dict[Hashable, List[Task]]) -> Dict[Hashable, List[Task]]:
    """Converts a schedule estimate to a schedule actual.
    
    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.
        schedule_estimate (Dict[Hashable, List[Task]]): The estimated schedule.

    Returns:
        Dict[Hashable, List[Task]]: The actual schedule.
    """
    schedule_actual = {
        node: [] for node in network.nodes
    }


class OnlineHeftScheduler(Scheduler):
    def __init__(self):
        self.heft_scheduler = HeftScheduler()

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        # Assume network and task graph node/edge attributes are:
        #   - weight_actual: float
        #   - weight_estimate: float

        schedule_actual: Dict[Hashable, List[Task]] = {
            node: [] for node in network.nodes
        }
        while True:
            # Step 1: set weight attributes to weight_actual if the task is in the schedule
            #         otherwise, set to weight_estimate

            schedule_estimate = self.heft_scheduler.schedule(network, task_graph, schedule_actual)
            schedule_actual_hypothetical = schedule_estimate_to_actual(network, task_graph, schedule_estimate)

            # Step 2: Get next task to finish based on *actual* execution time
            # Step 3: Add task to schedule_actual
            # Step 4: Repeat until there are no more tasks to schedule


        
def main():
    pass
