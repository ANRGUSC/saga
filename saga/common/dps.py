import logging
import pathlib
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np

from ..base import Task
from ..base import Scheduler
from ..utils.tools import check_instance_simple, get_insert_loc

thisdir = pathlib.Path(__file__).resolve().parent


def compute_tl(task, pred_tl):
    if len(task['predecessors']) == 0:
        return 0
    elif any(pred_tl[pred] is None for pred in task['predecessors']):
        return float('inf')
    else:
        return max(pred_tl[pred] + task['cost'][pred] for pred in task['predecessors'])


def compute_bl(task, succ_bl):
    if len(task['successors']) == 0:
        return 0
    elif any(succ_bl[succ] is None for succ in task['successors']):
        return float('inf')
    else:
        return max(succ_bl[succ] + task['cost'][task['processor']] + task['data_transfer'][succ] for succ in task['successors'])


def compute_eft(task, pred_eft):
    if len(task['predecessors']) == 0:
        return task['cost'][task['processor']]
    else:
        return max(pred_eft[pred] + task['cost'][task['processor']] + task['data_transfer'][pred] for pred in task['predecessors'])

def dps_rank_sort(network: nx.Graph, task_graph: nx.DiGraph) -> List[Hashable]:
    """Sort tasks based on their rank (as defined in the DPS paper).
    
    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        List[Hashable]: The sorted list of tasks.
    """
    tasks = list(task_graph.nodes())
    processors = list(network.nodes())

    tl = [None] * n
    bl = [None] * n
    priority = [None] * n
    eft = [None] * n
    ready_list = []

    n = len(tasks)
    m = len(processors)

    tl = [None] * n
    bl = [None] * n
    priority = [None] * n
    eft = [None] * n
    ready_list = []

    for i in range(n):
        tl[i] = compute_tl(tasks[i], tl)
        bl[i] = compute_bl(tasks[i], bl)
        priority[i] = bl[i] - tl[i]

    task_list = sorted(range(n), key=lambda x: priority[x], reverse=True)

    def update_ready_list():
        nonlocal ready_list
        ready_list = [task for task in ready_list if task in task_list]

    ready_list = task_list.copy()
    schedule = []

    while ready_list:
        task_index = ready_list[0]
        task = tasks[task_index]

        processor = min(range(m), key=lambda x: compute_eft(
            task, eft + [compute_eft(task, eft)]) if x != task['processor'] else float('inf'))

        schedule.append((task['id'], processors[processor]))
        ready_list.remove(task_index)

        for i in range(n):
            if tasks[i]['id'] != task['id'] and tasks[i]['processor'] == task['processor']:
                tl[i] = compute_tl(tasks[i], tl + [tl[task_index]])
                priority[i] = bl[i] - tl[i]

        update_ready_list()

        eft[task_index] = compute_eft(task, eft + [compute_eft(task, eft)])
        bl[task_index] = compute_bl(task, bl + [compute_bl(task, bl)])

    return schedule


class DpsSchedukler(Scheduler):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_runtimes(network: nx.Graph, task_graph: nx.DiGraph) -> Tuple[Dict[Hashable, Dict[Hashable, float]], Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]]:
        """Get the expected runtimes of all tasks on all nodes.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Tuple[Dict[Hashable, Dict[Hashable, float]], Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]]: A tuple of dictionaries mapping nodes to a dictionary of tasks and their runtimes and edges to a dictionary of tasks and their communication times.
                The first dictionary maps nodes to a dictionary of tasks and their runtimes.
                The second dictionary maps edges to a dictionary of task dependencies and their communication times.
        """
        runtimes = {}
        for node in network.nodes:
            runtimes[node] = {}
            speed: float = network.nodes[node]["weight"]
            for task in task_graph.nodes:
                cost: float = task_graph.nodes[task]["weight"]
                runtimes[node][task] = cost / speed
                logging.debug(
                    f"Task {task} on node {node} has runtime {runtimes[node][task]}")

        commtimes = {}
        for src, dst in network.edges:
            commtimes[src, dst] = {}
            commtimes[dst, src] = {}
            speed: float = network.edges[src, dst]["weight"]
            for src_task, dst_task in task_graph.edges:
                cost = task_graph.edges[src_task, dst_task]["weight"]
                commtimes[src, dst][src_task, dst_task] = cost / speed
                commtimes[dst, src][src_task, dst_task] = cost / speed
                logging.debug(
                    f"Task {src_task} on node {src} to task {dst_task} on node {dst} has communication time {commtimes[src, dst][src_task, dst_task]}")

        return runtimes, commtimes

    def _schedule(self,
                  network: nx.Graph,
                  task_graph: nx.DiGraph,
                  runtimes: Dict[Hashable, Dict[Hashable, float]],
                  commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
                  schedule_order: List[Hashable]) -> Dict[Hashable, List[Task]]:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            runtimes (Dict[Hashable, Dict[Hashable, float]]): A dictionary mapping nodes to a dictionary of tasks and their runtimes.
            commtimes (Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]): A dictionary mapping edges to a dictionary of task dependencies and their communication times.
            schedule_order (List[Hashable]): The order in which to schedule the tasks.

        Returns:
            Dict[Hashable, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """
        comp_schedule: Dict[Hashable, List[Task]] = {
            node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}

        task_name: Hashable
        logging.debug(f"Schedule order: {schedule_order}")
        for task_name in schedule_order:
            min_finish_time = np.inf
            best_node = None
            for node in network.nodes:  # Find the best node to run the task
                logging.debug(f"Testing task {task_name} on node {node}")
                max_arrival_time: float = max(
                    [
                        0.0, *[
                            task_schedule[parent].end + (
                                commtimes[(task_schedule[parent].node, node)][(
                                    parent, task_name)]
                            )
                            for parent in task_graph.predecessors(task_name)
                        ]
                    ]
                )

                runtime = runtimes[node][task_name]
                idx, start_time = get_insert_loc(
                    comp_schedule[node], max_arrival_time, runtime)

                finish_time = start_time + runtime
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_node = node, idx

            new_runtime = runtimes[best_node[0]][task_name]
            task = Task(best_node[0], task_name,
                        min_finish_time - new_runtime, min_finish_time)
            comp_schedule[best_node[0]].insert(best_node[1], task)
            task_schedule[task_name] = task

        return comp_schedule
