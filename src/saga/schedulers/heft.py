import pathlib
from typing import Dict,Hashable, List, Optional
import numpy as np
from copy import deepcopy
from queue import PriorityQueue


from saga import NetworkNode, Schedule, Scheduler, ScheduledTask, TaskGraph, Network
from saga.schedulers.cpop import upward_rank
from saga.utils.duplication import should_duplicate


thisdir = pathlib.Path(__file__).resolve().parent


def heft_rank_sort(network: Network, task_graph: TaskGraph) -> List[str]:
    """Sort tasks based on their rank (as defined in the HEFT paper).

    Args:
        network (Network): The network graph.
        task_graph (TaskGraph): The task graph.

    Returns:
        List[str]: The sorted list of tasks.
    """
    urank = upward_rank(network, task_graph)
    topological_sort = {
        node.name: i for i, node in enumerate(reversed(task_graph.topological_sort()))
    }
    rank = {node: (urank[node], topological_sort[node]) for node in urank}
    order = sorted(list(rank.keys()), key=lambda x: rank.get(x, 0.0), reverse=True)
    return order


class HeftScheduler(Scheduler):
    """Schedules tasks using the HEFT algorithm.

    Source: https://dx.doi.org/10.1109/71.993206
    """
    duplication_factor: int = 1

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,

        
    ) -> Schedule:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (Optional[Schedule], optional): The schedule. Defaults to None.
            min_start_time (float, optional): The minimum start time. Defaults to 0.0.

        Returns:
            Schedule: The schedule.
        Raises:
            ValueError: If the instance is invalid.
        """
        schedule_order = heft_rank_sort(network, task_graph)
        schedule = Schedule(task_graph, network)
        for task_name in schedule_order:
            task = task_graph.get_task(task_name)

            if schedule.is_scheduled(task_name):
                continue

            duplicate_factor = 1 
            if should_duplicate(task_name, task_graph, network):
                duplicate_factor = min(self.duplication_factor, len(task_graph.out_edges(task_name)))

            min_finish_time = np.inf
            best_nodes = PriorityQueue()
            for node in network.nodes:
                start_time = schedule.get_earliest_start_time(
                    task=task, node=node, append_only=False
                )
                start_time = max(start_time, min_start_time)
                runtime = (
                    task_graph.get_task(task_name).cost / network.get_node(node).speed
                )
                finish_time = start_time + runtime
                best_nodes.put((finish_time, node))


            for _ in range(duplicate_factor):
                if best_nodes.empty():
                    break
                min_finish_time: float
                best_node: NetworkNode
                min_finish_time, best_node = best_nodes.get()
                
                new_task = ScheduledTask(
                    node=best_node.name,
                    name=task_name,
                    start=min_finish_time
                    - (
                        task_graph.get_task(task_name).cost
                        / best_node.speed
                    ),
                    end=min_finish_time,
                )
                schedule.add_task(new_task)

        return schedule
