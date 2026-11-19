from queue import PriorityQueue
import pathlib
from typing import Any, List, Optional
import numpy as np


from saga import Schedule, Scheduler, ScheduledTask, TaskGraph, Network
from saga.schedulers.cpop import upward_rank


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

    With ``duplication_factor > 1``, a communication-heavy non-critical task is
    placed on up to that many nodes so its successors can read a local copy
    instead of paying the transfer cost. Critical-path tasks are never duplicated.
    """

    duplication_factor: int = 1
    # maps each task to their target processors
    # example: {"A": ["P1", "P2"]}
    duplication_targets: dict[str, list[str]] = {}

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

            target_nodes = set(self.duplication_targets.get(task_name, []))

            min_finish_time = np.inf
            best_nodes: PriorityQueue[Any] = PriorityQueue()
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

            scheduled_nodes = set()
            # always schedule the original task on the best node
            if not best_nodes.empty():
                min_finish_time, best_node = best_nodes.get()

                new_task = ScheduledTask(
                    node=best_node.name,
                    name=task_name,
                    start=min_finish_time
                    - (task_graph.get_task(task_name).cost / best_node.speed),
                    end=min_finish_time,
                )
                schedule.add_task(new_task)
                scheduled_nodes.add(best_node.name)

            # if beneficial target processors are found for this task,
            # create duplicate copies only on those beneficial processors
            # this prevents duplication from being placed on processors where it's unlikely to help/no children got placed here
            if target_nodes:
                target_candidates: PriorityQueue[Any] = PriorityQueue()

                for node in network.nodes:
                    if node.name not in target_nodes or node.name in scheduled_nodes:
                        continue

                    start_time = schedule.get_earliest_start_time(
                        task=task, node=node, append_only=False
                    )
                    start_time = max(start_time, min_start_time)
                    runtime = (
                        task_graph.get_task(task_name).cost
                        / network.get_node(node).speed
                    )
                    finish_time = start_time + runtime
                    target_candidates.put((finish_time, node))

                max_duplicates = self.duplication_factor - 1
                duplicates_added = 0

                while (
                    not target_candidates.empty() and duplicates_added < max_duplicates
                ):
                    # add duplicate copies in EFT order among the target processors
                    min_finish_time, best_node = target_candidates.get()

                    new_task = ScheduledTask(
                        node=best_node.name,
                        name=task_name,
                        start=min_finish_time
                        - (task_graph.get_task(task_name).cost / best_node.speed),
                        end=min_finish_time,
                    )

                    schedule.add_task(new_task)
                    scheduled_nodes.add(best_node.name)
                    duplicates_added += 1

        return schedule
