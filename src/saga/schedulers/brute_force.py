import itertools
from typing import Dict

import networkx as nx
import networkx.algorithms.dag as dag
from saga.scheduler import Scheduler, ScheduledTask, Schedule


class BruteForceScheduler(Scheduler):
    """Brute force scheduler"""
    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> Schedule:
        """Returns the best schedule (minimizing makespan) for a problem
           instance using brute force

        Args:
            network: Network
            task_graph: Task graph

        Returns:
            A dictionary of the schedule
        """
        # get all topological sorts of the task graph
        topological_sorts = list(dag.all_topological_sorts(task_graph))
        # get all valid mappings of the task graph nodes to the network nodes
        mappings = [
            dict(zip(task_graph.nodes, mapping))
            for mapping in itertools.product(network.nodes, repeat=len(task_graph.nodes))
        ]

        best_schedule = None
        best_makespan = float("inf")
        for mapping in mappings:
            for top_sort in topological_sorts:
                tasks: Dict[int, ScheduledTask] = {}
                schedule: Schedule = Schedule.create(network.nodes)
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
                    tasks[task] = ScheduledTask(node=node, name=task, start=ready_time, end=end_time)
                    schedule.add_task(tasks[task])

                if schedule.makespan < best_makespan:
                    best_makespan = schedule.makespan
                    best_schedule = schedule

        if best_schedule is None:
            raise RuntimeError("Brute force scheduler failed to find a schedule")
        return best_schedule