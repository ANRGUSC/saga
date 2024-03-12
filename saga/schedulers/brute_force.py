import itertools
from typing import Dict, Hashable, List

import networkx as nx

from ..scheduler import Scheduler, Task


class BruteForceScheduler(Scheduler):
    """Brute force scheduler"""
    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Returns the best schedule (minimizing makespan) for a problem
           instance using brute force

        Args:
            network: Network
            task_graph: Task graph

        Returns:
            A dictionary of the schedule
        """
        # get all topological sorts of the task graph
        topological_sorts = list(nx.algorithms.dag.all_topological_sorts(task_graph))
        # get all valid mappings of the task graph nodes to the network nodes
        mappings = [
            dict(zip(task_graph.nodes, mapping))
            for mapping in itertools.product(network.nodes, repeat=len(task_graph.nodes))
        ]

        best_schedule = None
        best_makespan = float("inf")
        for mapping in mappings:
            for top_sort in topological_sorts:
                tasks: Dict[int, Task] = {}
                schedule: Dict[int, List[Task]] = {}
                for task in top_sort:
                    node = mapping[task]
                    task_cost = task_graph.nodes[task]["weight"]
                    # get parents finish times + xfer time
                    ready_time = 0 if task_graph.in_degree(task) == 0 else max([
                        tasks[parent].end + (
                            task_graph[parent][task]["weight"] / network[mapping[parent]][node]["weight"]
                        ) for parent in task_graph.predecessors(task)
                    ])
                    # if node already has a task, get last tasks end time
                    if node in schedule:
                        ready_time = max(ready_time, schedule[node][-1].end)

                    node_speed = network.nodes[node]["weight"]
                    end_time = ready_time + task_cost / node_speed
                    tasks[task] = Task(node, task, ready_time, end_time)
                    schedule.setdefault(node, []).append(tasks[task])

                makespan = max([max([task.end for task in tasks]) for tasks in schedule.values() if len(tasks) > 0])
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_schedule = schedule

        # fill empty nodes with empty lists
        best_schedule = {node: best_schedule.get(node, []) for node in network.nodes}
        return best_schedule
