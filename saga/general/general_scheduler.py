from typing import Dict, Hashable, List, Callable
import networkx as nx

from .ScheduleType.schedule_type import ScheduleType
from ..scheduler import Scheduler, Task
from .utils import get_runtimes
class GeneralScheduler(Scheduler):
    """
    A general scheduler that takes in a ranking heuristic, a schedule type and a task selector and schedules a task graph onto a network.
    """
    def __init__(self, ranking_heuristic: Callable[[nx.Graph, nx.DiGraph], List[Hashable]], 
                 schedule_type: ScheduleType, 
                 selector: Callable[[nx.Graph, List[Hashable], ScheduleType], Dict[Hashable, List[Task]]]
                 ) -> None:
        """
        Args:
            ranking_heuristic (Callable[[nx.Graph, nx.DiGraph], List[Hashable]]): The heuristic used to get the initial ranking of the tasks
            schedule_type (ScheduleType): The type of schedule to use, for eg. insertion or append-only
            selector (Callable[[nx.Graph, List[Hashable], ScheduleType], Dict[Hashable, List[Task]]]): The task selector to use, for eg. HEFT or CPOP
        """
        self.ranking_heauristic = ranking_heuristic
        self.schedule_type = schedule_type
        self.selector = selector

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph
    ) -> Dict[Hashable, List[Task]]:
        """
        Schedule a task graph onto a network by parameters given in the constructor.

        Args:
            network (nx.Graph): The network to schedule onto.
            task_graph (nx.DiGraph): The task graph to schedule.

        Returns:
            Dict[Hashable, List[Task]]: The schedule.
        """
        rankings = self.ranking_heauristic(network, task_graph)
        runtimes, commtimes = get_runtimes(network, task_graph)
        schedule_type_obj = self.schedule_type(task_graph, runtimes, commtimes)
        return self.selector(network, rankings, schedule_type_obj)


