from abc import ABC, abstractmethod
import networkx as nx
from typing import Dict, Hashable, List, Set, Tuple
from saga.general.RankingHeuristics import UpwardRankSort
import saga.general.general_scheduler
from .earliest_finish_time import EarliestFinishTimeInsert
from saga.scheduler import Task
from .utils import insert
import numpy as np
import copy


def get_k_depth_children(
    task_graph: nx.DiGraph, task_name: Hashable, k: int, parents: Set
) -> [nx.DiGraph, Set]:
    new_task_graph = nx.DiGraph()
    new_task_graph.add_node(task_name)
    new_task_graph.nodes[task_name]["weight"] = task_graph.nodes[task_name]["weight"]
    parents.add(task_name)
    def recurse(task_name, k):
        if k == 0:
            return
        for child in task_graph.successors(task_name):
            if set(task_graph.predecessors(child)).issubset(parents):
                if child not in new_task_graph.nodes:
                    new_task_graph.add_node(child)
                    new_task_graph.nodes[child]["weight"] = task_graph.nodes[child][
                        "weight"
                    ]
                    parents.add(child)
                if (task_name, child) not in new_task_graph.edges:
                    new_task_graph.add_edge(task_name, child)
                    new_task_graph.edges[task_name, child]["weight"] = task_graph.edges[
                        task_name, child
                    ]["weight"]
                
                recurse(child, k - 1)

    recurse(task_name, k)
    return new_task_graph


class LookAheadInsert(ABC):
    def __init__(self, k=1, scheduler=None) -> None:
        self.k = k
        if scheduler is None:
            self.scheduler = saga.general.general_scheduler.GeneralScheduler(
                UpwardRankSort(), None, None, EarliestFinishTimeInsert()
            )
        else:
            self.scheduler = scheduler

    def __call__(
        self,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        runtimes: Dict[Hashable, Dict[Hashable, float]],
        commtimes: Dict[
            Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]
        ],
        comp_schedule: Dict[Hashable, List[Task]],
        task_schedule: Dict[Hashable, Task],
        task_name: Hashable,
        priority: int,
    ) -> None:
        if task_graph.out_degree(task_name) == 0:
            self.scheduler.insert_task(
                network,
                task_graph,
                runtimes,
                commtimes,
                comp_schedule,
                task_schedule,
                task_name,
                priority,
            )
            return

        min_finish_time = np.inf
        best_node = None
        _task_graph = get_k_depth_children(task_graph, task_name, self.k, set(task_schedule.keys()))
        for node in network.nodes:
            _task_schedule = copy.deepcopy(task_schedule)
            _comp_schedule = copy.deepcopy(comp_schedule)
            insert(
                task_graph,
                runtimes,
                commtimes,
                node,
                task_name,
                _comp_schedule,
                _task_schedule,
            )
            _comp_schedule = self.scheduler.schedule(
                network,
                _task_graph,
                _comp_schedule,
                _task_schedule,
                runtimes=runtimes,
                commtimes=commtimes,
                #Add a way to pass CPOP priorities here
            )
            finish_time = max(
                _task_schedule[task_name].end for task_name in _task_graph.nodes
            )

            if finish_time < min_finish_time:
                min_finish_time = finish_time
                best_node = node
        insert(
            task_graph,
            runtimes,
            commtimes,
            best_node,
            task_name,
            comp_schedule,
            task_schedule,
        )
