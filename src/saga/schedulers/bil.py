import copy
from typing import Dict, List, Optional, Set, Tuple

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph


class BILScheduler(Scheduler):  # pylint: disable=too-few-public-methods
    """Best Imaginary Level Scheduler

    Source: https://doi.org/10.1007/BFb0024750
    Modifications:
    - The original algorithm does not consider heterogenous communication strengths between
      network nodes. This affects the ipc_overhead term in the BIL definition. We compute the
      ipc_overhead between two tasks by scaling the cost by the average communication weight
      in the network.
        Original: BIL(task, node) = task_cost / node_speed + max(BIL(child, node) + ipc_overhead)
                                                                 for child in children(task))
    """

    def schedule(
        self,  # pylint: disable=too-many-locals
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Returns the schedule for the given task graph on the given network using the BIL algorithm.

        Args:
            network: Network
            task_graph: Task graph
            schedule: Optional initial schedule to build upon. Defaults to None.
            min_start_time: Minimum start time for tasks. Defaults to 0.0.

        Returns:
            A Schedule object containing the computed schedule.
        """
        comp_schedule = Schedule(task_graph, network)
        scheduled_tasks: Dict[str, ScheduledTask] = {}

        if schedule is not None:
            comp_schedule = schedule.model_copy()
            scheduled_tasks = {
                task.name: task for _, tasks in schedule.items() for task in tasks
            }

        bils: Dict[Tuple[str, str], float] = {}
        for task in reversed(task_graph.topological_sort()):
            for node in network.nodes:
                exec_time = task.cost / node.speed
                children_bils = []
                for out_edge in task_graph.out_edges(task.name):
                    child_name = out_edge.target
                    child_bil_same_node = bils[(child_name, node.name)]
                    other_node_bils = [
                        bils[(child_name, other_node.name)]
                        + (
                            out_edge.size
                            / network.get_edge(other_node.name, node.name).speed
                        )
                        for other_node in network.nodes
                    ]
                    children_bils.append(min(child_bil_same_node, *other_node_bils))
                bils[(task.name, node.name)] = exec_time + max(children_bils, default=0)

        ready_tasks: Set[str] = {
            task.name
            for task in task_graph.tasks
            if task_graph.in_degree(task.name) == 0
        }

        while len(scheduled_tasks) < len(list(task_graph.tasks)):
            # Section 3.1: Node Selection
            bims: Dict[str, List[Tuple[str, float]]] = {
                task_name: sorted(
                    [
                        (
                            node.name,
                            (
                                comp_schedule[node.name][-1].end
                                if comp_schedule[node.name]
                                else 0
                            )
                            + bils[(task_name, node.name)],
                        )
                        for node in network.nodes
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )
                for task_name in ready_tasks
                if task_name not in scheduled_tasks
            }

            j = 0
            selected_tasks = copy.copy(ready_tasks)
            while j < len(list(network.nodes)):
                max_bim = -1.0
                selected_tasks = set()
                for task_name, _bims in bims.items():
                    _, bim = _bims[j]
                    if bim == max_bim:
                        selected_tasks.add(task_name)
                    elif bim > max_bim:
                        max_bim = bim
                        selected_tasks = {task_name}
                if len(selected_tasks) == 1:
                    break
                j += 1

            selected_task_name = selected_tasks.pop()
            selected_task = task_graph.get_task(selected_task_name)

            # Section 3.2: Processor Selection
            # compute revised bims for selected task and all nodes
            revised_bims: Dict[str, float] = {
                node.name: (
                    (
                        comp_schedule[node.name][-1].end
                        if comp_schedule[node.name]
                        else 0
                    )
                    + bils[(selected_task_name, node.name)]
                    + (
                        selected_task.cost
                        / node.speed
                        * max(len(ready_tasks) / len(list(task_graph.tasks)) - 1, 0)
                    )
                )
                for node in network.nodes
            }

            # select node with lowest revised bim
            selected_node_name = min(
                revised_bims, key=lambda node_name: revised_bims[node_name]
            )
            # If more than one processor have the same revised BIM value, we select the
            # processor that makes the sum of the revised BIM values of other nodes on the
            # processor maximum.
            if (
                len(
                    [
                        node
                        for node in network.nodes
                        if revised_bims[node.name] == revised_bims[selected_node_name]
                    ]
                )
                > 1
            ):
                selected_node_name = max(
                    [node.name for node in network.nodes],
                    key=lambda node_name: sum(
                        revised_bims[other_node.name]
                        for other_node in network.nodes
                        if other_node.name != node_name
                    ),
                )

            selected_node = network.get_node(selected_node_name)

            # Schedule
            start_time = max(
                min_start_time,
                comp_schedule[selected_node_name][-1].end
                if comp_schedule[selected_node_name]
                else 0,
                max(
                    (
                        scheduled_tasks[in_edge.source].end
                        + (
                            in_edge.size
                            / network.get_edge(
                                selected_node_name, scheduled_tasks[in_edge.source].node
                            ).speed
                        )
                        for in_edge in task_graph.in_edges(selected_task_name)
                    ),
                    default=0,
                ),
            )
            end_time = start_time + selected_task.cost / selected_node.speed
            new_task = ScheduledTask(
                node=selected_node_name,
                name=selected_task_name,
                start=start_time,
                end=end_time,
            )
            comp_schedule.add_task(new_task)
            scheduled_tasks[selected_task_name] = new_task

            # Update ready tasks
            ready_tasks.update(
                {
                    out_edge.target
                    for out_edge in task_graph.out_edges(selected_task_name)
                    if all(
                        in_edge.source in scheduled_tasks
                        for in_edge in task_graph.in_edges(out_edge.target)
                    )
                }
            )

        return comp_schedule
