from typing import Dict, Hashable, List, Set, Tuple
import networkx as nx
from saga.scheduler import Task
import saga.general.general_scheduler
from saga.general.RankingHeuristics import UpwardRankSort
from saga.general.InsertTask import EarliestFinishTimeInsert
import numpy as np
import copy
from .tie_breaker import TieBreaker


def get_k_depth_graph(
    task_graph: nx.DiGraph, priority_queue: List, k: int, parent: Set
) -> nx.DiGraph:
    new_task_graph = nx.DiGraph()
    task_set_for_eval: List[Hashable] = []

    def recurse(task_name, k):
        if k == 0:
            return
        for child in task_graph.successors(task_name):
            if set(task_graph.predecessors(child)).issubset(parent):
                if child not in new_task_graph.nodes:
                    new_task_graph.add_node(child)
                    new_task_graph.nodes[child]["weight"] = task_graph.nodes[child][
                        "weight"
                    ]
                    task_set_for_eval.append(child)
                    parent.add(child)
                if (task_name, child) not in new_task_graph.edges:
                    new_task_graph.add_edge(task_name, child)
                    new_task_graph.edges[task_name, child]["weight"] = task_graph.edges[
                        task_name, child
                    ]["weight"]
                for pred in task_graph.predecessors(
                    child
                ):  # Add parents as the scheduler will look at them for scheduling
                    if pred not in new_task_graph.nodes:
                        new_task_graph.add_node(pred)
                        new_task_graph.nodes[pred]["weight"] = task_graph.nodes[pred][
                            "weight"
                        ]
                    if (pred, child) not in new_task_graph.edges:
                        new_task_graph.add_edge(pred, child)
                        new_task_graph.edges[pred, child]["weight"] = task_graph.edges[
                            pred, child
                        ]["weight"]
                recurse(child, k - 1)

    for task_name, _ in priority_queue:
        new_task_graph.add_node(task_name)
        new_task_graph.nodes[task_name]["weight"] = task_graph.nodes[task_name]["weight"]
        for pred in task_graph.predecessors(
            task_name
        ):  # Add parents as the scheduler will look at them for scheduling
            if pred not in new_task_graph.nodes:
                new_task_graph.add_node(pred)
                new_task_graph.nodes[pred]["weight"] = task_graph.nodes[pred]["weight"]
            if (pred, task_name) not in new_task_graph.edges:
                new_task_graph.add_edge(pred, task_name)
                new_task_graph.edges[pred, task_name]["weight"] = task_graph.edges[
                    pred, task_name
                ]["weight"]
        task_set_for_eval.append(task_name)
        parent.add(task_name)
        recurse(task_name, k)
    return new_task_graph, task_set_for_eval


class KDepthTieBreaker(TieBreaker):
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
        priority_queue: List,
    ) -> Tuple[Hashable, int]:
        min_finish_time = np.inf
        best_task, best_priority = None, None
        _task_graph, task_set = get_k_depth_graph(
            task_graph, priority_queue, self.k, set(task_schedule.keys())
        )
        for task_name, priority in priority_queue:
            _task_schedule = copy.deepcopy(task_schedule)
            _comp_schedule = copy.deepcopy(comp_schedule)
            self.scheduler.insert_task(
                network,
                _task_graph,
                runtimes,
                commtimes,
                _comp_schedule,
                _task_schedule,
                task_name,
                priority=priority,
            )
            _comp_schedule = self.scheduler.schedule(
                network,
                _task_graph,
                _comp_schedule,
                _task_schedule,
                runtimes=runtimes,
                commtimes=commtimes,
                # Add a way to pass CPOP priorities here
            )
            finish_time = max([_task_schedule[task_name].end for task_name in task_set])
            if finish_time < min_finish_time:
                min_finish_time = finish_time
                best_task = task_name
                best_priority = priority
        # print("Best task: ", best_task, "Best priority: ", best_priority)
        return (best_task, best_priority)
