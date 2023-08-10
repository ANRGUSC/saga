import random
from typing import Dict, Hashable, List

import networkx as nx

from ..base import Scheduler, Task


class WBAScheduler(Scheduler):
    """Worst-Case Bound Aware Scheduler"""
    def __init__(self, alpha: float = 0.5) -> None:
        super(WBAScheduler, self).__init__()
        self.alpha = alpha

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:


        schedule: Dict[Hashable, List[Task]] = {}
        scheduled_tasks: Dict[Hashable, Task] = {} # Map from task_name to Task

        def get_eet(task: Hashable, node: Hashable) -> float:
            return task_graph.nodes[task]['weight'] / network.nodes[node]['weight']

        def get_commtime(task1: Hashable, task2: Hashable, node1: Hashable, node2: Hashable) -> float:
            return task_graph.edges[task1, task2]['weight'] / network.edges[node1, node2]['weight']

        def get_eat(node: Hashable) -> float:
            return schedule[node][-1].end if schedule.get(node) else 0

        def get_fat(task: Hashable, node: Hashable) -> float:
            if task_graph.in_degree(task) <= 0:
                return 0
            return max(
                scheduled_tasks[pred_task].end +
                get_commtime(pred_task, task, scheduled_tasks[pred_task].node, node)
                for pred_task in task_graph.predecessors(task)
            )

        def get_ect(task: Hashable, node: Hashable) -> float:
            return get_eet(task, node) + max(get_eat(node), get_fat(task, node))

        while len(scheduled_tasks) < task_graph.order():
            available_tasks = [
                task for task in task_graph.nodes
                if (task not in scheduled_tasks and 
                    set(task_graph.predecessors(task)).issubset(set(scheduled_tasks.keys())))
            ]

            #implementation from paper
            while available_tasks:
                avail_pairs = []
                i_min = float('inf')
                i_max = -float('inf')

                for task in available_tasks:
                    for node in network.nodes:
                        i = get_ect(task, node) - get_eat(node)
                        if i <= i_min + self.alpha * (i_max - i_min):
                            avail_pairs.append((task, node))
                        i_min = min(i_min, i)
                        i_max = max(i_max, i)

                sched_task, sched_node = random.choice(avail_pairs)
                schedule.setdefault(sched_node, [])
                new_task = Task(
                    node=sched_node,
                    name=sched_task,
                    start=get_eat(sched_node),
                    end=get_ect(sched_task, sched_node)
                )
                schedule[sched_node].append(new_task)
                scheduled_tasks[sched_task] = new_task
                print(f"Task {sched_task} scheduled on machine {sched_node}")
                available_tasks.remove(sched_task)

        return schedule
