from itertools import product
from typing import Dict, Hashable, List

import networkx as nx

from ..base import Scheduler, Task


class FCPScheduler(Scheduler):
    """Schedules all tasks on the node with the highest processing speed"""
    def critical_path(self, task_graph: nx.DiGraph) -> List[Hashable]:
        """Returns the critical path of a task graph

        Args:
            task_graph: Task graph

        Returns:
            A list of tasks in the critical path
        """
        distance = {node: (0, []) for node in task_graph.nodes}

        for node in nx.topological_sort(task_graph):
            # We're only interested in the longest path
            if not distance[node][1]:
                score, path = 0, []
            else:
                score, path = max(
                    [(distance[node][0], path + [node]) if node in path else (0, []) for path in distance[node][1]],
                    key = lambda x: x[0]
                )
            for successor in task_graph.successors(node):
                if distance[successor][0] < score + task_graph.edges[node, successor]['weight']:
                    distance[successor] = (score + task_graph.edges[node, successor]['weight'], path)
        node, (_, path) = max(distance.items(), key=lambda x: x[1][0])

        return path

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Returns the best schedule (minimizing makespan) for a problem instance using FCP(Fastest Critical Path)

        Args:
            network: Network
            task_graph: Task graph

        Returns:
            A dictionary of the schedule
        """


        schedule: Dict[Hashable, List[Task]] = {}
        scheduled_tasks: Dict[Hashable, Task] = {} # Map from task_name to Task

        def get_eet(task: Hashable, node: Hashable) -> float:
            return task_graph.nodes[task]['weight'] / network.nodes[node]['weight']

        def get_commtime(task1: Hashable, task2: Hashable, node1: Hashable, node2: Hashable) -> float:
            return task_graph.edges[task1, task2]['weight'] / network.edges[node1, node2]['weight']

        # EAT = {node: 0 for node in network.nodes}
        def get_eat(node: Hashable) -> float:
            eat = schedule[node][-1].end if schedule.get(node) else 0
            return eat

        # FAT = np.zeros((num_tasks, num_machines))
        def get_fat(task: Hashable, node: Hashable) -> float:
            fat = 0 if task_graph.in_degree(task) <= 0 else max([
                # schedule[r_schedule[pred_task]][-1].end +
                scheduled_tasks[pred_task].end +
                get_commtime(pred_task, task, scheduled_tasks[pred_task].node, node)
                for pred_task in task_graph.predecessors(task)
            ])
            return fat

        def get_ect(task: Hashable, node: Hashable) -> float:
            return get_eet(task, node) + max(get_eat(node), get_fat(task, node))

        # Get the critical path
        critical_path = self.critical_path(task_graph)

        all_tasks = list(task_graph.nodes)
        ready_tasks = [task for task in all_tasks if task_graph.in_degree(task) == 0]

        while ready_tasks:
            critical_ready_tasks = [task for task in ready_tasks if task in critical_path]
            non_critical_ready_tasks = [task for task in ready_tasks if task not in critical_path]

            if critical_ready_tasks:
                sched_task, sched_node = min(
                    product(critical_ready_tasks, network.nodes),
                    key=lambda instance: get_ect(instance[0], instance[1])
                )
            else:
                sched_task, sched_node = min(
                    product(non_critical_ready_tasks, network.nodes),
                    key=lambda instance: get_ect(instance[0], instance[1])
                )

            schedule.setdefault(sched_node, [])
            new_task = Task(
                node=sched_node,
                name=sched_task,
                start=max(get_eat(sched_node), get_fat(sched_task, sched_node)),
                end=get_ect(sched_task, sched_node)
            )
            schedule[sched_node].append(new_task)
            scheduled_tasks[sched_task] = new_task
            ready_tasks.remove(sched_task)

            for succ_task in task_graph.successors(sched_task):
                if all(pred_task in scheduled_tasks for pred_task in task_graph.predecessors(succ_task)):
                    ready_tasks.append(succ_task)

        return schedule
