import logging
from typing import Dict, Optional, Tuple
import numpy as np

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph


def calc_TEC(
    task_name: str, network: Network, task_graph: TaskGraph, mode: str = "avg"
) -> float:
    match mode:
        case "avg":
            return avg_TEC(task_name, network, task_graph)
        case "median":
            return median_TEC(task_name, network, task_graph)
        case "max":
            return max_TEC(task_name, network, task_graph)
        case "min":
            return min_TEC(task_name, network, task_graph)

    raise ValueError(f"Invalid mode: {mode}")


# max Task Execution Cost
def max_TEC(task_name: str, network: Network, task_graph: TaskGraph) -> float:
    task = task_graph.get_task(task_name)
    return max(task.cost / node.speed for node in network.nodes)


# avg Task Execution Cost
def avg_TEC(task_name: str, network: Network, task_graph: TaskGraph) -> float:
    task = task_graph.get_task(task_name)
    return sum(task.cost / node.speed for node in network.nodes) / len(
        list(network.nodes)
    )


# median Task Execution Cost
def median_TEC(task_name: str, network: Network, task_graph: TaskGraph) -> float:
    task = task_graph.get_task(task_name)
    return float(np.median([task.cost / node.speed for node in network.nodes]))


# min Task Execution Cost
def min_TEC(task_name: str, network: Network, task_graph: TaskGraph) -> float:
    task = task_graph.get_task(task_name)
    return min(task.cost / node.speed for node in network.nodes)


def calc_TL(
    task_name: str,
    network: Network,
    task_graph: TaskGraph,
    assigned_tasks: Dict[str, str],
) -> float:
    in_edges = task_graph.in_edges(task_name)
    if not in_edges:
        return 0
    max_TL = 0.0
    for in_edge in in_edges:
        pred = in_edge.source
        # This is the first term of the equation
        TL = calc_TL(pred, network, task_graph, assigned_tasks)

        # This is the second term of the equation
        if assigned_tasks.get(pred) is None:
            TL += calc_TEC(pred, network, task_graph)
        else:
            pred_task = task_graph.get_task(pred)
            pred_node = network.get_node(assigned_tasks[pred])
            TL += pred_task.cost / pred_node.speed

        # This is the third term of the equation
        if assigned_tasks.get(pred) is None or assigned_tasks.get(task_name) is None:
            TL += in_edge.size
        else:
            network_edge = network.get_edge(
                assigned_tasks[pred], assigned_tasks[task_name]
            )
            TL += in_edge.size / network_edge.speed

        if TL > max_TL:
            max_TL = TL
    return max_TL


def calc_BL(
    task_name: str,
    network: Network,
    task_graph: TaskGraph,
    assigned_tasks: Dict[str, str],
) -> float:
    out_edges = task_graph.out_edges(task_name)
    if not out_edges:
        if assigned_tasks.get(task_name) is None:
            return calc_TEC(task_name, network, task_graph)
        else:
            task = task_graph.get_task(task_name)
            node = network.get_node(assigned_tasks[task_name])
            return task.cost / node.speed
    max_BL = 0.0
    for out_edge in out_edges:
        succ = out_edge.target
        # This is the first term of the equation
        BL = calc_BL(succ, network, task_graph, assigned_tasks)

        # This is the second term of the equation
        if assigned_tasks.get(succ) is None or assigned_tasks.get(task_name) is None:
            BL += out_edge.size
        else:
            network_edge = network.get_edge(
                assigned_tasks[task_name], assigned_tasks[succ]
            )
            BL += out_edge.size / network_edge.speed

        # This is the third term of the equation
        if assigned_tasks.get(task_name) is None:
            BL += calc_TEC(task_name, network, task_graph)
        else:
            task = task_graph.get_task(task_name)
            node = network.get_node(assigned_tasks[task_name])
            BL += task.cost / node.speed

        if BL > max_BL:
            max_BL = BL
    return max_BL


def calc_priority(
    task_name: str,
    network: Network,
    task_graph: TaskGraph,
    assigned_tasks: Dict[str, str],
) -> float:
    return calc_BL(task_name, network, task_graph, assigned_tasks) - calc_TL(
        task_name, network, task_graph, assigned_tasks
    )


class DPSScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_runtimes(
        network: Network, task_graph: TaskGraph
    ) -> Tuple[
        Dict[str, Dict[str, float]], Dict[Tuple[str, str], Dict[Tuple[str, str], float]]
    ]:
        """Get the expected runtimes of all tasks on all nodes.

        Args:
            network (Network): The network.
            task_graph (TaskGraph): The task graph.

        Returns:
            Tuple[Dict[str, Dict[str, float]], Dict[Tuple[str, str], Dict[Tuple[str, str], float]]]: A tuple of dictionaries mapping nodes to a dictionary of tasks and their runtimes and edges to a dictionary of tasks and their communication times.
                The first dictionary maps nodes to a dictionary of tasks and their runtimes.
                The second dictionary maps edges to a dictionary of task dependencies and their communication times.
        """
        runtimes: Dict[str, Dict[str, float]] = {}
        for node in network.nodes:
            runtimes[node.name] = {}
            for task in task_graph.tasks:
                runtimes[node.name][task.name] = task.cost / node.speed
                logging.debug(
                    f"Task {task.name} on node {node.name} has runtime {runtimes[node.name][task.name]}"
                )

        commtimes: Dict[Tuple[str, str], Dict[Tuple[str, str], float]] = {}
        for edge in network.edges:
            src, dst = edge.source, edge.target
            if (src, dst) not in commtimes:
                commtimes[src, dst] = {}
            if (dst, src) not in commtimes:
                commtimes[dst, src] = {}
            for dep in task_graph.dependencies:
                commtimes[src, dst][dep.source, dep.target] = dep.size / edge.speed
                commtimes[dst, src][dep.source, dep.target] = dep.size / edge.speed
                logging.debug(
                    f"Task {dep.source} on node {src} to task {dep.target} on node {dst} has communication time {commtimes[src, dst][dep.source, dep.target]}"
                )

        return runtimes, commtimes

    def _schedule(
        self,
        task_graph: TaskGraph,
        network: Network,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        task_list = [task.name for task in task_graph.topological_sort()]
        assigned_tasks: Dict[str, str] = {}
        task_schedule: Dict[str, ScheduledTask] = {}

        comp_schedule = Schedule(task_graph, network)
        if schedule is not None:
            comp_schedule = schedule.model_copy()
            task_schedule = {
                task.name: task for _, tasks in schedule.items() for task in tasks
            }
            assigned_tasks = {task.name: task.node for task in task_schedule.values()}

        task_list = [t for t in task_list if t not in task_schedule]
        task_list.sort(
            key=lambda x: calc_priority(x, network, task_graph, assigned_tasks),
            reverse=True,
        )
        ready_list = task_list.copy()
        runtimes, commtimes = DPSScheduler.get_runtimes(network, task_graph)

        while len(ready_list) > 0:
            task_name = ready_list.pop(0)
            # Earliest Finish Time
            min_finish_time = np.inf
            best_node = next(iter(network.nodes)).name  # arbitrary initialization
            for node in network.nodes:
                logging.debug(f"Trying to assign task {task_name} to node {node.name}")
                runtime = runtimes[node.name][task_name]
                start_time = comp_schedule.get_earliest_start_time(
                    task=task_name, node=node.name, append_only=True
                )

                finish_time = start_time + runtime
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_node = node.name

            new_runtime = runtimes[best_node][task_name]
            task_ob = ScheduledTask(
                node=best_node,
                name=task_name,
                start=min_finish_time - new_runtime,
                end=min_finish_time,
            )
            comp_schedule.add_task(task_ob)
            task_schedule[task_name] = task_ob
            assigned_tasks[task_name] = best_node
            ready_list.sort(
                key=lambda x: calc_priority(x, network, task_graph, assigned_tasks),
                reverse=True,
            )
        return comp_schedule

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        return self._schedule(task_graph, network, schedule, min_start_time)
