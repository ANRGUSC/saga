from functools import lru_cache
import random
from typing import Dict, Hashable, List, Tuple

import networkx as nx

from ....scheduler import Scheduler, Task


class ResidualWBAScheduler(Scheduler): # pylint: disable=too-few-public-methods
    """Workflow-Based Allocation (WBA) scheduler.
    
    Source: http://dx.doi.org/10.1109/CCGRID.2005.1558639
    """
    def __init__(self, alpha: float = 0.5) -> None:
        """Initializes the WBA scheduler.

        Args:
            alpha (float, optional): The alpha parameter. Defaults to 0.5.
        """
        super(ResidualWBAScheduler, self).__init__()
        self.alpha = alpha

    def schedule(self, network: nx.Graph, task_graphs: List[Tuple[nx.DiGraph, float]]) -> Dict[Hashable, List[Task]]:
        """Returns the schedule of the given task graph on the given network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: The schedule.
        """
        schedule: Dict[Hashable, List[Task]] = {}
        scheduled_tasks: Dict[Hashable, Task] = {} # Map from task_name to Task

        for task_graph_tupple in task_graphs:
            task_graph = task_graph_tupple[0]
            task_graph_arrival_time = task_graph_tupple[1]


            @lru_cache(maxsize=None)
            def get_eet(task: Hashable, node: Hashable) -> float:
                """Estimated execution time of task on node"""
                return task_graph.nodes[task]['weight'] / network.nodes[node]['weight']

            @lru_cache(maxsize=None)
            def get_commtime(task1: Hashable, task2: Hashable, node1: Hashable, node2: Hashable) -> float:
                """Communication time between task1 and task2 on node1 and node2"""
                return task_graph.edges[task1, task2]['weight'] / network.edges[node1, node2]['weight']

            @lru_cache(maxsize=None) # Must clear cache after each iteration since schedule changes
            def get_eat(node: Hashable) -> float:
                """Earliest available time of node"""
                return schedule[node][-1].end if schedule.get(node) else 0

            @lru_cache(maxsize=None) # Must clear cache after each iteration since schedule changes
            def get_fat(task: Hashable, node: Hashable) -> float:
                """Latest available time of task on node"""
                if task_graph.in_degree(task) <= 0:
                    return task_graph_arrival_time
                return max(
                    scheduled_tasks[pred_task].end +
                    get_commtime(pred_task, task, scheduled_tasks[pred_task].node, node)
                    for pred_task in task_graph.predecessors(task)
                )

            @lru_cache(maxsize=None) # Must clear cache after each iteration since schedule changes
            def get_est(task: Hashable, node: Hashable) -> float:
                """Earliest start time of task on node"""
                return max(get_eat(node), get_fat(task, node))

            @lru_cache(maxsize=None) # Must clear cache after each iteration since schedule changes
            def get_ect(task: Hashable, node: Hashable) -> float:
                """Earliest completion time of task on node"""
                return get_eet(task, node) + get_est(task, node)
            
            def clear_caches():
                """Clear all caches."""
                get_eat.cache_clear()
                get_fat.cache_clear()
                get_est.cache_clear()
                get_ect.cache_clear()

            cur_makespan = 0
            # while len(scheduled_tasks) < task_graph.order():
            while set(task_graph.nodes).issubset(set(scheduled_tasks.keys())) == False:
                available_tasks = [
                    task for task in task_graph.nodes
                    if (task not in scheduled_tasks and
                        set(task_graph.predecessors(task)).issubset(set(scheduled_tasks.keys())))
                ]

                while available_tasks:
                    i_min = float('inf')
                    i_max = -float('inf')

                    makespan_increases: Dict[Hashable, Tuple[Hashable, Hashable]] = {}
                    for task in available_tasks:
                        for node in network.nodes:
                            makespan_increases[task, node] = max(get_ect(task, node) - cur_makespan, 0) # makespan increase

                    i_min = min(makespan_increases.values())
                    i_max = max(makespan_increases.values())
                    avail_pairs = [
                        key for key, value in makespan_increases.items()
                        if value <= i_min + self.alpha * (i_max - i_min)
                    ]

                    sched_task, sched_node = random.choice(avail_pairs)
                    schedule.setdefault(sched_node, [])
                    new_task = Task(
                        node=sched_node,
                        name=sched_task,
                        start=get_est(sched_task, sched_node),
                        end=get_ect(sched_task, sched_node)
                    )
                    schedule[sched_node].append(new_task)
                    scheduled_tasks[sched_task] = new_task
                    cur_makespan = max(cur_makespan, new_task.end)
                    available_tasks.remove(sched_task)

                    clear_caches()
        # fill empty nodes with empty list
        for node in network.nodes:
            if node not in schedule:
                schedule[node] = []
        return schedule