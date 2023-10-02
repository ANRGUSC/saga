import logging
from typing import Dict, Hashable, List, Tuple
import networkx as nx
import numpy as np

from ..base import Scheduler, Task
from ..utils.tools import check_instance_simple, get_insert_loc

# options for mode can be 'avg', 'median', 'max' or 'min'
mode = 'avg'

def calc_TEC() -> float:
    match mode:
        case 'avg':
            return avg_TEC()
        case 'median':
            return median_TEC()
        case 'max':
            return max_TEC()
        case 'min':
            return min_TEC()

def max_TEC(task: Hashable, network: nx.Graph, task_graph: nx.DiGraph) -> float:
    return max(task_graph.nodes[task]['weight'] / network.nodes[node]['weight'] for node in network.nodes)

def avg_TEC(task: Hashable, network: nx.Graph, task_graph: nx.DiGraph) -> float:
    return sum(task_graph.nodes[task]['weight'] / network.nodes[node]['weight'] for node in network.nodes) / len(network.nodes)

def median_TEC(task: Hashable, network: nx.Graph, task_graph: nx.DiGraph) -> float:
    return np.median(task_graph.nodes[task]['weight'] / network.nodes[node]['weight'] for node in network.nodes)

def min_TEC(task: Hashable, network: nx.Graph, task_graph: nx.DiGraph) -> float:
    return min(task_graph.nodes[task]['weight'] / network.nodes[node]['weight'] for node in network.nodes)  

def calc_TL(task: Hashable, network: nx.Graph, task_graph: nx.DiGraph, assigned_tasks: dict) -> float:
    if(len(task_graph.predecessors(task_graph.nodes[task]))) == 0:
        return 0
    max_TL = 0
    for pred in task_graph.predecessors(task_graph.nodes[task]):
        "This is the first term of the equation"
        TL = calc_TL(pred, network, task_graph)

        "This is the second term of the equation"
        if assigned_tasks.get(pred) is None:
            TL += calc_TEC(pred, network, task_graph)
        else:
            TL += task_graph.nodes[pred]['weight'] / network.nodes[assigned_tasks.get(pred)]['weight']

        "This is the third term of the equation"
        if assigned_tasks.get(pred) is None or assigned_tasks.get(task) is None:
            TL += task_graph.edges[pred, task]['weight']
        elif assigned_tasks.get(pred) == assigned_tasks.get(task):
            TL += 0
        else:
            TL += task_graph.edges[pred, task]['weight'] / network.edges[assigned_tasks.get(pred), assigned_tasks.get(task)]['weight']
        
        if(TL > max_TL):
            max_TL = TL
    return max_TL

def calc_BL(task: Hashable, network: nx.Graph, task_graph: nx.DiGraph, assigned_tasks: dict) -> float:
    if(len(task_graph.successors(task_graph.nodes[task]))) == 0:
        if assigned_tasks.get(task) is None:
            return calc_TEC(task, network, task_graph)
        else:
            return task_graph.nodes[task]['weight'] / network.nodes[assigned_tasks.get(task)]['weight']
    max_BL = 0
    for succ in task_graph.successors(task_graph.nodes[task]):
        "This is the first term of the equation"
        BL = calc_BL(succ, network, task_graph)

        "This is the second term of the equation"
        if assigned_tasks.get(succ) is None or assigned_tasks.get(task) is None:
            BL += task_graph.edges[task, succ]['weight']
        elif assigned_tasks.get(succ) == assigned_tasks.get(task):
            BL += 0
        else:
            BL += task_graph.edges[task, succ]['weight'] / network.edges[assigned_tasks.get(task), assigned_tasks.get(succ)]['weight']

        "This is the third term of the equation"
        if assigned_tasks.get(task) is None:
            BL += calc_TEC(task, network, task_graph)
        else:
            BL += task_graph.nodes[task]['weight'] / network.nodes[assigned_tasks.get(task)]['weight']

        if(BL > max_BL):
            max_BL = BL
    return max_BL
    
def calc_priority(task: Hashable, network: nx.Graph, task_graph: nx.DiGraph, assigned_tasks: dict) -> float:
    return calc_TL(task, network, task_graph, assigned_tasks) + calc_BL(task, network, task_graph, assigned_tasks)
    
class DpsScheduler(Scheduler):
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
                logging.debug(f"Task {task} on node {node} has runtime {runtimes[node][task]}")

        commtimes = {}
        for src, dst in network.edges:
            commtimes[src, dst] = {}
            commtimes[dst, src] = {}
            speed: float = network.edges[src, dst]["weight"]
            for src_task, dst_task in task_graph.edges:
                cost = task_graph.edges[src_task, dst_task]["weight"]
                commtimes[src, dst][src_task, dst_task] = cost / speed
                commtimes[dst, src][src_task, dst_task] = cost / speed
                logging.debug(f"Task {src_task} on node {src} to task {dst_task} on node {dst} has communication time {commtimes[src, dst][src_task, dst_task]}")

        return runtimes, commtimes

    def _schedule(self, task_graph: nx.DiGraph, network: nx.Graph) -> Dict[str, List[Task]]:
        task_list = list(task_graph.nodes)
        task_list.sort(key=lambda x: calc_priority(x, network, task_graph), reverse=True)
        ready_list = task_list.copy()
        assigned_tasks = {}
        comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}
        runtimes, commtimes = DpsScheduler.get_runtimes(network, task_graph)

        while len(ready_list) > 0:
            task = ready_list.pop(0)
            "Earliest Finish Time"
            min_finish_time = np.inf
            best_node = None
            for node in network.nodes:
                logging.debug(f"Trying to assign task {task} to node {node}")
                max_arrival_time: float = max( # 
                    [
                        0.0, *[
                            task_schedule[parent].end + (
                                commtimes[(task_schedule[parent].node, node)][(parent, task_name)]
                            )
                            for parent in task_graph.predecessors(task)
                        ]
                    ]
                )

                runtime = runtimes[node][task]   
                idx, start_time = get_insert_loc(comp_schedule[node], max_arrival_time, runtime)
                
                finish_time = start_time + runtime
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_node = node, idx 

            new_runtime = runtimes[best_node[0]][task]
            task = Task(best_node[0], task, min_finish_time - new_runtime, min_finish_time)
            comp_schedule[best_node[0]].insert(best_node[1], task)
            task_schedule[task] = task
            assigned_tasks[task] = best_node[0]
            ready_list.sort(key=lambda x: calc_priority(x, network, task_graph, assigned_tasks), reverse=True)
        return comp_schedule

    def schedule(self, task_graph: nx.DiGraph, network: nx.Graph) -> Dict[str, List[Task]]:
        check_instance_simple(network, task_graph)
        return self._schedule(task_graph, network)