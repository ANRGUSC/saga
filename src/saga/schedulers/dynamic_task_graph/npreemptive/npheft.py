import logging
import pathlib
from typing import Dict, Hashable, List, Tuple, Optional

import networkx as nx
import numpy as np
from saga.utils.draw import draw_gantt

from ....scheduler import Task, DWScheduler
from ....utils.tools import get_insert_loc
from ...cpop import upward_rank

thisdir = pathlib.Path(__file__).resolve().parent


def heft_rank_sort(network: nx.Graph, task_graph: nx.DiGraph) -> List[Hashable]:
    rank = upward_rank(network, task_graph)
    topological_sort = {node: i for i, node in enumerate(reversed(list(nx.topological_sort(task_graph))))}
    rank = {node: (rank[node] + topological_sort[node]) for node in rank}
    return sorted(list(rank.keys()), key=rank.get, reverse=True)


class NPHeftScheduler(DWScheduler):

    @staticmethod
    def get_runtimes(
        network: nx.Graph, task_graph: nx.DiGraph
    ) -> Tuple[
        Dict[Hashable, Dict[Hashable, float]],
        Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],]:
        
        runtimes = {}
        for node in network.nodes:
            runtimes[node] = {}
            speed: float = network.nodes[node]["weight"]
            for task in task_graph.nodes:
                cost: float = task_graph.nodes[task]["weight"]
                runtimes[node][task] = cost / speed
                logging.debug(
                    "Task %s on node %s has runtime %s",
                    task,
                    node,
                    runtimes[node][task],
                )

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
                    "Task %s on node %s to task %s on node %s has communication time %s",
                    src_task,
                    src,
                    dst_task,
                    dst,
                    commtimes[src, dst][src_task, dst_task],
                )

        return runtimes, commtimes

    def _schedule(
        self,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        runtimes: Dict[Hashable, Dict[Hashable, float]],
        commtimes: Dict[
            Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]
        ],
        schedule_order: List[Hashable],
        current_schedule: Optional[Dict[str, List[Task]]] = None,
        task_graph_arrival_time: Optional[float] = 0
        ) -> Dict[Hashable, List[Task]]:


        comp_schedule: Dict[Hashable, List[Task]] = current_schedule or {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}

        task_name: Hashable
        logging.debug("Schedule order: %s", schedule_order)
        for task_name in schedule_order:

            min_finish_time = np.inf
            best_node = None
            for node in network.nodes:  # Find the best node to run the task 
                max_arrival_time: float = max(
                    [
                        task_graph_arrival_time,
                        *[
                            task_schedule[parent].end
                            + (
                                commtimes[(task_schedule[parent].node, node)][
                                    (parent, task_name)
                                ]
                            )
                            for parent in task_graph.predecessors(task_name)
                        ],
                    ]
                )

                runtime = runtimes[node][task_name]
                idx, start_time = get_insert_loc(
                    comp_schedule[node], max_arrival_time, runtime
                )

                logging.debug(
                    "Testing task %s on node %s: start time %s, finish time %s",
                    task_name,
                    node,
                    start_time,
                    start_time + runtime,
                )

                finish_time = start_time + runtime
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_node = node, idx

            new_runtime = runtimes[best_node[0]][task_name]
            task = Task(
                best_node[0], task_name, min_finish_time - new_runtime, min_finish_time
            )
            comp_schedule[best_node[0]].insert(best_node[1], task)
            task_schedule[task_name] = task

        return comp_schedule

    def schedule(
        self, network: nx.Graph, task_graphs: List[Tuple[nx.DiGraph, float]]
    ) -> Dict[str, List[Task]]:
        
        comp_schedule: Dict[Hashable, List[Task]] = None

        for task_graph_tupple in task_graphs:
            task_graph = task_graph_tupple[0]
            task_graph_arrival_time = task_graph_tupple[1]
            runtimes, commtimes = NPHeftScheduler.get_runtimes(network, task_graph)
            schedule_order = heft_rank_sort(network, task_graph)
            comp_schedule = self._schedule(
                network, task_graph, runtimes, commtimes, schedule_order, comp_schedule, task_graph_arrival_time
            )

        return comp_schedule
