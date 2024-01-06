from queue import PriorityQueue
from abc import abstractmethod
from typing import Dict, Hashable, List, Callable, Optional, Tuple
import networkx as nx
from ..scheduler import Scheduler, Task
from .utils import get_runtimes

# QUESTIONS: Are Prioritqueue and ranking_heuristic the same thing for dynamic scheduling?


class Filter:
    @abstractmethod
    def __call__(
        self,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        priority_queue: PriorityQueue,
        schedule: Dict[Hashable, List[Task]],
    ) -> PriorityQueue:
        raise NotImplementedError


class GeneralScheduler(Scheduler):
    """
    A general scheduler that takes in a ranking heuristic, a schedule type and a task selector and schedules a task graph onto a network.
    """

    def __init__(
        self,
        ranking_heuristic: Callable[[nx.Graph, nx.DiGraph], List],
        tie_breaker: Callable[
            [
                nx.Graph,
                nx.DiGraph,
                Dict[Hashable, Dict[Hashable, float]],
                Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
                Dict[Hashable, List[Task]],  # comp_schedule
                Dict[Hashable, Task],  # task_schedule
                Optional[List],  # PriorityQueue, need to enforce types here
            ],
            Hashable,
        ],
        # filter: Filter,
        insert_task: Callable[
            [
                nx.Graph,
                nx.DiGraph,
                Dict[Hashable, Dict[Hashable, float]],  # runtimes
                Dict[
                    Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]
                ],  # commtimes
                Dict[Hashable, List[Task]],  # comp_schedule
                Dict[Hashable, Task],  # task_schedule
                Hashable,
            ],  # task_name
            None,
        ],
        k=1,
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
                lambda network, task_graph, runtimes, commtimes, comp_schedule, task_schedule, priority_queue: priority_queue[0]
            )
        self.insert_task = insert_task
        self.k = k

    def schedule(
        self, network: nx.Graph, task_graph: nx.DiGraph
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
        while priority_queue:
            # Apply Filter here
            # ready_tasks = []

            ready_tasks = []

            for task_name, priority in priority_queue:
                if set(task_graph.predecessors(task_name)).issubset(
                    task_schedule.keys()
                ):
                    ready_tasks.append((task_name, priority))

            task_name, priority = self.tie_breaker(
                network, task_graph, runtimes, commtimes, comp_schedule, task_schedule, ready_tasks
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
            )

            priority_queue.remove((task_name, priority))

        return comp_schedule
