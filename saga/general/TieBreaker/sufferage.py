from typing import Dict, Hashable, List, Tuple
import networkx as nx
from saga.scheduler import Task
from .tie_breaker import TieBreaker


class Sufferage(TieBreaker):
    def __init__(self):
        pass

    def __call__(
        self,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        runtimes: Dict[Hashable, Dict[Hashable, float]],
        commtimes: Dict[
            Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]
        ],
        comp_schedule: Dict[Hashable, List[Task]],
        task_schedule: Dict[Hashable, Task],
        priority_queue: List,
        rankings:List
    ) -> Tuple[Hashable, int]:
        def get_eat(node: Hashable) -> float:
            """Earliest available time on a node"""
            return comp_schedule[node][-1].end if comp_schedule.get(node) else 0

        def get_fat(task: Hashable, node: Hashable) -> float:
            """Get file availability time of a task on a node"""
            return (
                0
                if task_graph.in_degree(task) <= 0
                else max(
                    [
                        task_schedule[pred_task].end +
                        # get_commtime(pred_task, task, task_schedule[pred_task].node, node)
                        commtimes[task_schedule[pred_task].node, node][pred_task, task]
                        for pred_task in task_graph.predecessors(task)
                    ]
                )
            )

        def get_ect(task: Hashable, node: Hashable) -> float:
            """Get estimated completion time of a task on a node"""
            return runtimes[node][task] + max(get_eat(node), get_fat(task, node))

        sufferage_pq = {}
        for task, priority in priority_queue:
            ect_values = [get_ect(task, node) for node in network.nodes]
            first_ect = min(ect_values)
            ect_values.remove(first_ect)
            second_ect = min(ect_values) if ect_values else first_ect
            # sufferage_pq.put(-(second_ect - first_ect), (task, priority))
            sufferage_pq[(task, priority)] = second_ect - first_ect

        return max(sufferage_pq.keys(), key=lambda x: sufferage_pq[x])
