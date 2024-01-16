from queue import PriorityQueue
from abc import abstractmethod
from typing import Dict, Hashable, List, Callable, Optional, Tuple
import copy
import networkx as nx
from saga.general.InsertTask import InsertTask
from saga.general.TieBreaker import TieBreaker
from ..scheduler import Scheduler, Task
from .utils import get_runtimes
from saga.general.RankingHeuristics import RankingHeuristic
from saga.general.Filters import Filter


# class Filter:
#     @abstractmethod
#     def __call__(
#         self,
#         network: nx.Graph,
#         task_graph: nx.DiGraph,
#         priority_queue: PriorityQueue,
#         schedule: Dict[Hashable, List[Task]],
#     ) -> PriorityQueue:
#         raise NotImplementedError


class GeneralScheduler(Scheduler):
    """
    A general scheduler that takes in a ranking heuristic, a schedule type and a task selector and schedules a task graph onto a network.
    """

    def __init__(
        self,
        ranking_heuristic: RankingHeuristic,
        filter: Filter,
        tie_breaker: TieBreaker,
        insert_task: InsertTask,
        k:int =1,
    ) -> None:
        """
        Args:
            ranking_heuristic (Callable[[nx.Graph, nx.DiGraph], List[Hashable]]): The heuristic used to get the initial ranking of the tasks
            schedule_type (ScheduleType): The type of schedule to use, for eg. insertion or append-only
            selector (Callable[[nx.Graph, List[Hashable], ScheduleType], Dict[Hashable, List[Task]]]): The task selector to use, for eg. HEFT or CPOP
        """
        self.ranking_heauristic = ranking_heuristic
        if tie_breaker:
            self.tie_breaker = tie_breaker
        else:
            self.tie_breaker = (
                lambda network, task_graph, runtimes, commtimes, comp_schedule, task_schedule, ready_task, _: ready_task[ 0]
            )
        self.insert_task = insert_task
        self.k = k
        if filter:
            self.filter = filter
        else:
            self.filter = (
                lambda network, task_graph, priority_queue, schedule: priority_queue
            )

    def schedule(
        self, network: nx.Graph, task_graph: nx.DiGraph,
        comp_schedule: Dict[Hashable, List[Task]] = None,
        task_schedule: Dict[Hashable, Task] = None,
        k_steps=float("inf"),
        runtimes: Dict[Hashable, Dict[Hashable, float]] = None,
        commtimes: Dict[
            Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]
        ] = None,
        rankings =None
    ) -> Dict[Hashable, List[Task]]:
        """
        Schedule a task graph onto a network by parameters given in the constructor.

        Args:
            network (nx.Graph): The network to schedule onto.
            task_graph (nx.DiGraph): The task graph to schedule.

        Returns:
            Dict[Hashable, List[Task]]: The schedule.
        """
        if not rankings:
            priority_queue = self.ranking_heauristic(network, task_graph)
        else:
            priority_queue = copy.deepcopy(rankings)
        if runtimes is None or commtimes is None:
            assert runtimes is None and commtimes is None, "Both runtimes and commtimes must be None or neither must be None"
            runtimes, commtimes = get_runtimes(network, task_graph)

        if not comp_schedule:
            comp_schedule = {node: [] for node in network.nodes}
        if not task_schedule:
            task_schedule = task_schedule or {}
        k = 0
        i=0
        while i<len(priority_queue):
            task_name, priority = priority_queue[i]
            if task_name in task_schedule or task_name not in task_graph.nodes:
                priority_queue.remove((task_name, priority))
            else:
                i+=1
                
        while priority_queue and k < k_steps:

            ready_tasks = []

            for task_name, priority in priority_queue:
                if task_name in task_graph.nodes and set(task_graph.predecessors(task_name)).issubset(
                    task_schedule.keys()
                ):
                    ready_tasks.append((task_name, priority))
            
            ready_tasks = self.filter(
                network, task_graph, ready_tasks, comp_schedule
            )
            task_name, priority = self.tie_breaker(
                network, task_graph, runtimes, commtimes, comp_schedule, task_schedule, ready_tasks, priority_queue
            )

            self.insert_task(
                network,
                task_graph,
                runtimes,
                commtimes,
                comp_schedule,
                task_schedule,
                task_name,
                priority,
                priority_queue
            )

            priority_queue.remove((task_name, priority))
            k+=1

        return comp_schedule
