from queue import PriorityQueue
from abc import abstractmethod
from typing import Dict, Hashable, List, Callable, Optional, Tuple
import networkx as nx
from ..scheduler import Scheduler, Task
from .utils import get_runtimes
#QUESTIONS: Are Prioritqueue and ranking_heuristic the same thing for dynamic scheduling?

class Filter:
    @abstractmethod
    def __call__(self, network: nx.Graph, task_graph: nx.DiGraph, priority_queue: PriorityQueue, schedule: Dict[Hashable, List[Task]]) -> PriorityQueue:
        raise NotImplementedError



class GeneralScheduler(Scheduler):
    """
    A general scheduler that takes in a ranking heuristic, a schedule type and a task selector and schedules a task graph onto a network.
    """
    def __init__(self,
                ranking_heuristic: Callable[[nx.Graph, nx.DiGraph], List[Hashable]], 
                get_priority_queue: Callable[[nx.Graph, nx.DiGraph, Optional[PriorityQueue], Optional[Dict[Hashable, List[Task]]]], PriorityQueue],
                # filter: Filter,
                insert_task: Callable[[nx.Graph,
                                       nx.DiGraph,
                                       Dict[Hashable, Dict[Hashable, float]], #runtimes
                                       Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]], #commtimes
                                       Dict[Hashable, List[Task]], #comp_schedule
                                       Dict[Hashable, Task], #task_schedule
                                       Hashable], #task_name
                                       None],
                k = 1
                ) -> None:
        """
        Args:
            ranking_heuristic (Callable[[nx.Graph, nx.DiGraph], List[Hashable]]): The heuristic used to get the initial ranking of the tasks
            schedule_type (ScheduleType): The type of schedule to use, for eg. insertion or append-only
            selector (Callable[[nx.Graph, List[Hashable], ScheduleType], Dict[Hashable, List[Task]]]): The task selector to use, for eg. HEFT or CPOP
        """
        self.ranking_heauristic = ranking_heuristic
        if get_priority_queue:
            self.get_priority_queue = get_priority_queue
        else:
            self.get_priority_queue = lambda network, task_graph, priority_queue, schedule: priority_queue
        self.insert_task = insert_task
        self.k = k

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
        priority_queue = self.ranking_heauristic(network, task_graph)
        runtimes, commtimes = get_runtimes(network, task_graph)
        comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}
        priority_queue = self.get_priority_queue(network, task_graph, priority_queue, comp_schedule)
        while priority_queue.qsize()>0:
            
            
            # _pq = PriorityQueue([(task_name, priority) for priority, task_name in priority_queue.queue
            #                     if task_graph.prededecessors.issubset(task_schedule.keys())])

            ranking, (task_name, priority) = priority_queue.get()
            tempList = []
            while not set(task_graph.predecessors(task_name)).issubset(task_schedule.keys()):
                tempList.append((ranking, task_name, priority))
                ranking, (task_name, priority) = priority_queue.get()
            self.insert_task(network, task_graph, runtimes, commtimes, comp_schedule, task_schedule, task_name, priority)
            priority_queue = self.get_priority_queue(network, task_graph, priority_queue, comp_schedule)
            for ranking, task_name, priority in tempList:
                priority_queue.put((ranking, (task_name, priority)))

        return comp_schedule


