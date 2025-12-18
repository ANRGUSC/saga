from queue import PriorityQueue
from typing import Dict, List, Optional, Set

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph


def get_mcp_priorities(network: Network, task_graph: TaskGraph) -> Dict[str, float]:
    """Returns the priorities of the tasks on the network

    Args:
        network (Network): The network.
        task_graph (TaskGraph): The task graph.

    Returns:
        Dict[str, float]: The priorities of the tasks on the network.
    """
    avg_node_speed = sum(node.speed for node in network.nodes) / len(
        list(network.nodes)
    )

    # Filter edges: exclude self-loops unless there's only one node
    num_nodes = len(list(network.nodes))
    non_self_edges = [
        edge for edge in network.edges if edge.source != edge.target or num_nodes == 1
    ]
    avg_comm_speed = (
        sum(edge.speed for edge in non_self_edges) / len(non_self_edges)
        if non_self_edges
        else 1.0
    )

    # Build scaled task weights and edge weights
    scaled_task_weights: Dict[str, float] = {
        task.name: task.cost / avg_node_speed for task in task_graph.tasks
    }
    scaled_edge_weights: Dict[tuple, float] = {
        (dep.source, dep.target): dep.size / avg_comm_speed
        for dep in task_graph.dependencies
    }

    # Compute longest path lengths from each task to exit
    longest_path_lengths: Dict[str, float] = {}
    for task in reversed(task_graph.topological_sort()):
        avg_exec_time = scaled_task_weights[task.name]
        out_edges = task_graph.out_edges(task.name)
        if not out_edges:
            longest_path_lengths[task.name] = avg_exec_time
        else:
            longest_path_lengths[task.name] = avg_exec_time + max(
                longest_path_lengths[out_edge.target]
                + scaled_edge_weights[(out_edge.source, out_edge.target)]
                for out_edge in out_edges
            )

    critical_path_length = max(longest_path_lengths.values())
    # paths with greatest priority have the least critical path length - longest path length
    priorities = {
        task_name: critical_path_length - longest_path_lengths[task_name]
        for task_name in longest_path_lengths
    }
    return priorities


class FCPScheduler(Scheduler):
    """Fast Critical Path Scheduler

    Source: https://doi.org/10.1145/305138.305162
    Note: This original algorithm assumes the network communication/computation speeds are the same for all nodes.
    This implementation allows for different speeds by scaling the task weights by the average speeds (the algorithm
    will still perform poorly for heterogeneous networks, but it will at least produce valid schedules).
    """

    def __init__(self, priority_queue_size: Optional[int] = None):
        super().__init__()
        self.priority_queue_size = priority_queue_size

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Returns the best schedule (minimizing makespan) for a problem instance using FCP(Fastest Critical Path)

        Args:
            network: Network
            task_graph: Task graph
            schedule: Optional initial schedule. Defaults to None.
            min_start_time: Minimum start time. Defaults to 0.0.

        Returns:
            A Schedule object containing the computed schedule.
        """
        comp_schedule = Schedule(task_graph, network)
        scheduled_tasks: Dict[str, ScheduledTask] = {}

        if schedule is not None:
            comp_schedule = schedule.model_copy()
            scheduled_tasks = {
                t.name: t for _, tasks in schedule.items() for t in tasks
            }

        queue_priority: PriorityQueue = PriorityQueue(
            maxsize=self.priority_queue_size or len(list(network.nodes))
        )
        queue_fifo: List[str] = []
        queued_tasks: Set[str] = set()
        priorities = get_mcp_priorities(network, task_graph)

        def add_ready_task(task_name: str):
            if queue_priority.qsize() < queue_priority.maxsize:
                queue_priority.put((priorities[task_name], task_name))
            else:
                queue_fifo.append(task_name)
            queued_tasks.add(task_name)

        def select_ready_task() -> str:
            task_name = queue_priority.get()[1]
            if queue_fifo:
                fifo_task = queue_fifo.pop(0)
                queue_priority.put((priorities[fifo_task], fifo_task))
            return task_name

        def get_exec_time(task_name: str, node_name: str) -> float:
            task = task_graph.get_task(task_name)
            node = network.get_node(node_name)
            return task.cost / node.speed

        def get_commtime(task1: str, task2: str, node1: str, node2: str) -> float:
            dep = task_graph.get_dependency(task1, task2)
            edge = network.get_edge(node1, node2)
            return dep.size / edge.speed

        def get_eat(node_name: str) -> float:
            tasks = comp_schedule[node_name]
            return tasks[-1].end if tasks else min_start_time

        def get_fat(task_name: str, node_name: str) -> float:
            in_edges = task_graph.in_edges(task_name)
            if not in_edges:
                return min_start_time
            return max(
                scheduled_tasks[in_edge.source].end
                + get_commtime(
                    in_edge.source,
                    task_name,
                    scheduled_tasks[in_edge.source].node,
                    node_name,
                )
                for in_edge in in_edges
            )

        def get_start_time(task_name: str, node_name: str) -> float:
            return max(get_eat(node_name), get_fat(task_name, node_name))

        def select_processor(task_name: str) -> str:
            # processor that becomes idle first
            p_start = min(
                network.nodes,
                key=lambda node: comp_schedule[node.name][-1].end
                if comp_schedule[node.name]
                else min_start_time,
            ).name
            # processor with predecessor that last finishes
            in_edges = task_graph.in_edges(task_name)
            if not in_edges:
                return p_start
            pred = max(
                [in_edge.source for in_edge in in_edges],
                key=lambda t: scheduled_tasks[t].end,
            )
            p_arrive = scheduled_tasks[pred].node

            if get_start_time(task_name, p_start) <= get_start_time(
                task_name, p_arrive
            ):
                return p_start
            return p_arrive

        for task in task_graph.tasks:
            if (
                task.name not in scheduled_tasks
                and task_graph.in_degree(task.name) == 0
            ):
                add_ready_task(task.name)

        num_tasks = len(list(task_graph.tasks))
        while len(scheduled_tasks) < num_tasks:
            task_name = select_ready_task()
            node_name = select_processor(task_name)
            start_time = get_start_time(task_name, node_name)
            exec_time = get_exec_time(task_name, node_name)
            new_task = ScheduledTask(
                node=node_name,
                name=task_name,
                start=start_time,
                end=start_time + exec_time,
            )
            scheduled_tasks[task_name] = new_task
            comp_schedule.add_task(new_task)

            for out_edge in task_graph.out_edges(task_name):
                succ = out_edge.target
                if all(
                    in_edge.source in scheduled_tasks
                    for in_edge in task_graph.in_edges(succ)
                ):
                    if succ not in queued_tasks:
                        add_ready_task(succ)

        return comp_schedule
