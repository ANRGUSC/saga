import logging
from typing import Dict, Hashable, List, Tuple
import networkx as nx

from .ScheduleType.schedule_type import ScheduleType
from ..scheduler import Scheduler, Task
from .utils import get_runtimes
class GeneralScheduler(Scheduler):
    def __init__(self, ranking_heuristic, schedule_type:ScheduleType, selector) -> None:
        self.ranking_heauristic = ranking_heuristic
        self.schedule_type = schedule_type
        self.selector = selector

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph
    ) -> Dict[Hashable, List[Task]]:
        rankings = self.ranking_heauristic(network, task_graph)
        runtimes, commtimes = get_runtimes(network, task_graph)
        schedule_type_obj = self.schedule_type(task_graph, runtimes, commtimes)
        return self.selector(network, rankings, schedule_type_obj)


