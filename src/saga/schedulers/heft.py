import pathlib
from typing import List, Optional

from saga import Schedule, Scheduler, ScheduledTask, TaskGraph, Network
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

    With ``duplication_factor > 1``, a communication-heavy task (one whose average
    outgoing communication cost exceeds its average computation cost) is placed on
    up to that many nodes, so its successors can read a local copy instead of paying
    the transfer cost. This can reduce makespan on communication-bound instances at
    the cost of extra compute.
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
            schedule (Optional[Schedule], optional): An existing partial schedule to
                extend. Defaults to None.
            min_start_time (float, optional): The minimum start time. Defaults to 0.0.

        Returns:
            Schedule: The resulting schedule.

        Raises:
            ValueError: If the instance is invalid.
        """
        schedule_order = heft_rank_sort(network, task_graph)
        schedule = schedule if schedule is not None else Schedule(task_graph, network)

        for task_name in schedule_order:
            if schedule.is_scheduled(task_name):
                continue
            task_cost = task_graph.get_task(task_name).cost
            # Finish time of this task on each node. Copies are independent (a copy's
            # start depends only on the task's parents), so compute this once for all.
            placements = []
            for node in network.nodes:
                start_time = schedule.get_earliest_start_time(
                    task=task_name,
                    node=node,
                    append_only=False,
                    current_moment=min_start_time,
                )
                finish_time = start_time + task_cost / network.get_node(node).speed
                placements.append((finish_time, node))
            placements.sort(key=lambda p: p[0])

            num_copies = 1
            if self.duplication_factor > 1 and should_duplicate(
                task_name, task_graph, network
            ):
                num_copies = max(
                    1,
                    min(
                        self.duplication_factor,
                        len(task_graph.out_edges(task_name)),
                        len(placements),
                    ),
                )

            for finish_time, node in placements[:num_copies]:
                schedule.add_task(
                    ScheduledTask(
                        node=node.name,
                        name=task_name,
                        start=finish_time - task_cost / network.get_node(node).speed,
                        end=finish_time,
                    )
                )

        return schedule
