from functools import lru_cache
from typing import Dict, Hashable, List, Tuple, Optional
import networkx as nx

from ....scheduler import Scheduler, Task, DWScheduler
from saga.utils.draw import draw_gantt


class PMaxMinScheduler(DWScheduler): # pylint: disable=too-few-public-methods
    """Max-Min scheduler"""
    def schedule(self,
                 network: nx.Graph,
                 task_graphs: List[Tuple[nx.DiGraph, float]]
                 ) -> Dict[Hashable, List[Task]]:
        """Schedules the task graph on the network

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: The schedule.
        """

        for task_graph_tupple in task_graphs:
            for node in task_graph_tupple[0].nodes:
                task_graph_tupple[0].nodes[node]["arrival_time"] = task_graph_tupple[1]

        

        schedule: Dict[Hashable, List[Task]] = {}
        scheduled_tasks: Dict[Hashable, Task] = {} # Map from task_name to Task

        for i in range(len(task_graphs)):
            task_graph = nx.compose_all([task_graphs[j][0] for j in range(i + 1)])


            if i > 0:
                for task_name in task_graph.nodes:
                    matching_task = next((task for tasks in schedule.values() for task in tasks if task.name == task_name), None)
                    if matching_task:
                        if matching_task.start > task_graphs[i][1]:
                            schedule[matching_task.node].remove(matching_task)
                            scheduled_tasks.pop(task_name, None)




            @lru_cache(maxsize=None)
            def get_eet(task: Hashable, node: Hashable) -> float:
                return task_graph.nodes[task]['weight'] / network.nodes[node]['weight']

            @lru_cache(maxsize=None)
            def get_commtime(task1: Hashable, task2: Hashable, node1: Hashable, node2: Hashable) -> float:
                return task_graph.edges[task1, task2]['weight'] / network.edges[node1, node2]['weight']

            @lru_cache(maxsize=None) # Must clear cache after each iteration since schedule changes
            def get_eat(node: Hashable,
                        task: Optional[Hashable] = None) -> float:
                return max(schedule[node][-1].end if schedule.get(node) else 0, task_graph.nodes[task]["arrival_time"])

            @lru_cache(maxsize=None) # Must clear cache after each iteration since schedule changes
            def get_fat(task: Hashable, node: Hashable) -> float:
                return task_graph.nodes[task]["arrival_time"] if task_graph.in_degree(task) <= 0 else max([
                    scheduled_tasks[pred_task].end +
                    get_commtime(pred_task, task, scheduled_tasks[pred_task].node, node)
                    for pred_task in task_graph.predecessors(task)
                ])

            @lru_cache(maxsize=None) # Must clear cache after each iteration since schedule changes
            def get_ect(task: Hashable, node: Hashable) -> float:
                return get_eet(task, node) + max(get_eat(node, task), get_fat(task, node))

            def clear_caches():
                """Clear all caches."""
                get_eat.cache_clear()
                get_fat.cache_clear()
                get_ect.cache_clear()

            # while len(scheduled_tasks) < task_graph.order():
            while set(task_graph.nodes).issubset(set(scheduled_tasks.keys())) == False:
                # get all tasks that are ready to be scheduled
                available_tasks = [
                    task for task in task_graph.nodes
                    if task not in scheduled_tasks and all(pred in scheduled_tasks for pred in task_graph.predecessors(task))
                ]
                #slight change from MinMin logic
                while available_tasks:
                    min_ects = {task: min(get_ect(task, node) for node in network.nodes) for task in available_tasks}
                    sched_task = max(min_ects, key=min_ects.get)
                    sched_node = min(network.nodes, key=lambda node: get_ect(sched_task, node))

                    schedule.setdefault(sched_node, [])
                    new_task = Task(
                        node=sched_node,
                        name=sched_task,
                        start=max(get_eat(sched_node, sched_task), get_fat(sched_task, sched_node)),
                        end=get_ect(sched_task, sched_node)
                    )
                    schedule[sched_node].append(new_task)
                    scheduled_tasks[sched_task] = new_task

                    available_tasks.remove(sched_task)

                    clear_caches()

        # add empty list for nodes that have no tasks scheduled
        for node in network.nodes:
            schedule.setdefault(node, [])
        
        # sort schedule pairs by key name
        schedule = dict(sorted(schedule.items()))

        return schedule
