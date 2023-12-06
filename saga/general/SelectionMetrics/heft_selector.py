from typing import Dict, Hashable, List, Tuple

import numpy as np
from saga.general.ScheduleType.schedule_type import ScheduleType
import networkx as nx

from saga.scheduler import Task


def heft_selector(network: nx.Graph,
        ranking: List[Hashable],
        scheduler : ScheduleType) -> Dict[Hashable, List[Task]]:
    
    comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
    task_schedule: Dict[Hashable, Task] = {}
    for task_name in ranking:
        min_finish_time = np.inf
        best_node = None
        for node in network.nodes:  # Find the best node to run the task
            finish_time = scheduler.get_earliest_finish(node, task_name, comp_schedule, task_schedule)
            if finish_time < min_finish_time:
                min_finish_time = finish_time
                best_node = node
        
        scheduler.insert(best_node, task_name, comp_schedule, task_schedule)
    
    return comp_schedule

