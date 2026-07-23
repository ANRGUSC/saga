from queue import PriorityQueue
from typing import Dict, Hashable, List, Optional, Set, Tuple

import networkx as nx

from ....scheduler import Scheduler, Task

def get_mcp_priorities(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """Returns the priorities of the tasks on the network

    Args:
        network (nx.Graph): The network.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        Dict[Hashable, float]: The priorities of the tasks on the network.
    """
    network = network.copy()
    task_graph = task_graph.copy()

    avg_node_speed = sum(
        network.nodes[node]['weight'] for node in network.nodes
    ) / len(network.nodes)
    avg_comm_speed = sum(
        network.edges[edge]['weight'] for edge in network.edges
        # if there is only one node, the avg is the self-loop edge weight
        if edge[0] != edge[1] or len(network.nodes) == 1
    ) / len(network.edges)

    # scale task weights by average speeds
    for task in task_graph.nodes:
        task_graph.nodes[task]['weight'] /= avg_node_speed
    for edge in task_graph.edges:
        task_graph.edges[edge]['weight'] /= avg_comm_speed

    # add dummy src and sink tasks
    src = '__mcp_src__'
    sink = '__mcp_sink__'
    task_graph.add_node(src, weight=1e-9)
    task_graph.add_node(sink, weight=1e-9)
    for task in task_graph.nodes:
        if task not in (src, sink) and task_graph.in_degree(task) == 0:
            task_graph.add_edge(src, task, weight=1e-9)
        if task not in (src, sink) and task_graph.out_degree(task) == 0:
            task_graph.add_edge(task, sink, weight=1e-9)

    longest_path_lengths = {}
    for task in reversed(list(nx.topological_sort(task_graph))):
        avg_exec_time = task_graph.nodes[task]['weight'] / avg_node_speed
        if task == sink:
            longest_path_lengths[task] = avg_exec_time
        else:
            longest_path_lengths[task] = avg_exec_time + max(
                longest_path_lengths[succ] + task_graph.edges[task, succ]['weight'] / avg_comm_speed
                for succ in task_graph.successors(task)
            )

    critical_path_length = max(longest_path_lengths.values())
    # paths with greatest priority have the least critical path length - longest path length
    priorities = {
        task: critical_path_length - longest_path_lengths[task]
        for task in task_graph.nodes
        if task not in (src, sink)
    }
    return priorities


class ResidualFCPScheduler(Scheduler): # pylint: disable=too-few-public-methods
    """Fast Critical Path Scheduler

    Source: https://doi.org/10.1145/305138.305162
    Note: This original algorithm assumes the network communication/computation speeds are the same for all nodes.
    This implementation allows for different speeds by scaling the task weights by the average speeds (the algorithm
    will still perform poorly for heterogeneous networks, but it will at least produce valid schedules).
    """
    def __init__(self, priority_queue_size: Optional[int] = None):
        super().__init__()
        self.priority_queue_size = priority_queue_size

    def schedule(self, network: nx.Graph, task_graphs: List[Tuple[nx.DiGraph, float]]) -> Dict[Hashable, List[Task]]:
        """Returns the best schedule (minimizing makespan) for a problem instance using FCP(Fastest Critical Path)

        Args:
            network: Network
            task_graph: Task graph

        Returns:
            A dictionary of the schedule
        """
        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        scheduled_tasks: Dict[Hashable, Task] = {} # Map from task_name to Task

        for task_graph_tupple in task_graphs:
            task_graph = task_graph_tupple[0]
            task_graph_arrival_time = task_graph_tupple[1]

            queue_priority = PriorityQueue(maxsize=self.priority_queue_size or len(network.nodes))
            queue_fifo = []
            queued_tasks: Set[Hashable] = set()
            priorities = get_mcp_priorities(network, task_graph)

            def add_ready_task(task: Hashable):
                if queue_priority.qsize() < queue_priority.maxsize:
                    queue_priority.put((priorities[task], task))
                else:
                    queue_fifo.append(task)
                queued_tasks.add(task)

            def select_ready_task() -> Hashable:
                task = queue_priority.get()[1]
                if queue_fifo:
                    fifo_task = queue_fifo.pop(0)
                    queue_priority.put((priorities[fifo_task], fifo_task))
                return task

            def get_exec_time(task: Hashable, node: Hashable) -> float:
                return task_graph.nodes[task]['weight'] / network.nodes[node]['weight']

            def get_commtime(task1: Hashable, task2: Hashable, node1: Hashable, node2: Hashable) -> float:
                return task_graph.edges[task1, task2]['weight'] / network.edges[node1, node2]['weight']

            def get_eat(node: Hashable) -> float:
                eat = schedule[node][-1].end if schedule.get(node) else task_graph_arrival_time
                return eat

            def get_fat(task: Hashable, node: Hashable) -> float:
                fat = task_graph_arrival_time if task_graph.in_degree(task) <= 0 else max(
                    scheduled_tasks[pred_task].end +
                    get_commtime(pred_task, task, scheduled_tasks[pred_task].node, node)
                    for pred_task in task_graph.predecessors(task)
                )
                return fat

            def get_start_time(task: Hashable, node: Hashable) -> float:
                return max(get_eat(node), get_fat(task, node))

            def select_processor(task: Hashable) -> Hashable:
                # processor that becomes idle first
                p_start = min(
                    network.nodes,
                    key=lambda node: schedule[node][-1].end if schedule[node] else task_graph_arrival_time
                )
                # processor with predecessor that last finishes
                predecessors: List[Hashable] = list(task_graph.predecessors(task))
                if not predecessors:
                    return p_start
                pred = max(
                    predecessors,
                    key=lambda task: scheduled_tasks[task].end
                )
                p_arrive = scheduled_tasks[pred].node

                if get_start_time(task, p_start) <= get_start_time(task, p_arrive):
                    return p_start
                return p_arrive

            for task in task_graph.nodes:
                if task_graph.in_degree(task) == 0:
                    add_ready_task(task)

            # while len(scheduled_tasks) < len(task_graph.nodes):
            while set(task_graph.nodes).issubset(set(scheduled_tasks.keys())) == False:
                task = select_ready_task()
                node = select_processor(task)
                start_time = get_start_time(task, node)
                exec_time = get_exec_time(task, node)
                scheduled_tasks[task] = Task(
                    node=node,
                    name=task,
                    start=start_time,
                    end=start_time+exec_time
                )
                schedule.setdefault(node, []).append(scheduled_tasks[task])
                for succ in task_graph.successors(task):
                    if all(pred in scheduled_tasks for pred in task_graph.predecessors(succ)):
                        if succ not in queued_tasks:
                            add_ready_task(succ)

        return schedule
