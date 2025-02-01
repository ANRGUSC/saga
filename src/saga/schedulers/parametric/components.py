from copy import deepcopy
import heapq

import numpy as np
from saga.scheduler import Scheduler, Task
from saga.schedulers.parametric import IntialPriority, ScheduleType, InsertTask, ParametricScheduler
from saga.schedulers.heft import heft_rank_sort, get_insert_loc
from saga.schedulers.cpop import cpop_ranks


import networkx as nx
from typing import Any, Dict, List, Hashable, Optional

# Initial Priority functions
class UpwardRanking(IntialPriority):
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> List[Hashable]:
        return heft_rank_sort(network, task_graph)
    
    def serialize(self) -> Dict[str, Any]:
        return {"name": "UpwardRanking"}
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "UpwardRanking":
        return cls()

class CPoPRanking(IntialPriority):
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> List[Hashable]:
        ranks = cpop_ranks(network, task_graph)
        start_task = max(
            [task for task in task_graph if task_graph.in_degree(task) == 0],
            key=ranks.get
        )
        pq = [(-ranks[start_task], start_task)]
        heapq.heapify(pq)
        queue = []
        while pq:
            _, task_name = heapq.heappop(pq)
            queue.append(task_name)
            ready_tasks = [
                succ for succ in task_graph.successors(task_name)
                if all(pred in queue for pred in task_graph.predecessors(succ))
            ]
            for ready_task in ready_tasks:
                heapq.heappush(pq, (-ranks[ready_task], ready_task))
        
        return queue
    
    def serialize(self) -> Dict[str, Any]:
        return {"name": "CPoPRanking"}
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "CPoPRanking":
        return cls()

class ArbitraryTopological(IntialPriority):
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> List[Hashable]:
        return list(nx.topological_sort(task_graph))
    
    def serialize(self) -> Dict[str, Any]:
        return {"name": "ArbitraryTopological"}
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "ArbitraryTopological":
        return cls()

GREEDY_INSERT_COMPARE_FUNCS = {
    "EFT": lambda new, cur: new.end - cur.end,
    "EST": lambda new, cur: new.start - cur.start,
    "Quickest": lambda new, cur: (new.end - new.start) - (cur.end - cur.start)
}
# Insert Task functions
class GreedyInsert(InsertTask):
    def __init__(self,
                 append_only: bool = False,
                 compare: str = "EFT",
                 critical_path: bool = False):
        """Initialize the GreedyInsert class.
        
        Args:
            append_only (bool, optional): Whether to only append the task to the schedule. Defaults to False.
            compare (Callable[[Task, Task], float], optional): The comparison function to use. Defaults to lambda new, cur: new.end - cur.end.
                Must be one of "EFT", "EST", or "Quickest".
            critical_path (bool, optional): Whether to only schedule tasks on the critical path. Defaults to False.
        """
        self.append_only = append_only
        self.compare = compare
        # self._compare = GREEDY_INSERT_COMPARE_FUNCS[compare]
        self.critical_path = critical_path

    def _compare(self, new: Task, cur: Task) -> float:
        return GREEDY_INSERT_COMPARE_FUNCS[self.compare](new, cur)

    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: ScheduleType,
                 task: Hashable,
                 node: Optional[Hashable] = None,
                 dry_run: bool = False) -> Task:
        best_insert_loc, best_task = None, None
        if self.critical_path and node is None:
            if "critical_path" not in task_graph.nodes[task]:
                ranks = cpop_ranks(network, task_graph)
                critical_rank = max(ranks.values())
                fastest_node = max(
                    network.nodes,
                    key=lambda node: network.nodes[node]['weight']
                )
                nx.set_node_attributes(
                    task_graph,
                    {
                        task: fastest_node if np.isclose(rank, critical_rank) else None
                     for task, rank in ranks.items()
                    },
                    "critical_path"
                )

            node = task_graph.nodes[task]["critical_path"]

        considered_nodes = network.nodes if node is None else [node]
        for node in considered_nodes:
            exec_time = task_graph.nodes[task]['weight'] / network.nodes[node]['weight']

            min_start_time = 0
            for parent in task_graph.predecessors(task):
                parent_task: Task = task_graph.nodes[parent]['scheduled_task']
                parent_node = parent_task.node
                data_size = task_graph.edges[parent, task]["weight"]
                comm_strength = network.edges[parent_node, node]["weight"]
                comm_time = data_size / comm_strength
                min_start_time = max(min_start_time, parent_task.end + comm_time)

            if self.append_only:
                start_time = max(
                    min_start_time,
                    0.0 if not schedule[node] else max(schedule[node][-1].end, 0.0)
                )
                insert_loc = len(schedule[node])
            else:
                insert_loc, start_time  = get_insert_loc(schedule[node], min_start_time, exec_time)

            new_task = Task(node, task, start_time, start_time + exec_time)
            if best_task is None or self._compare(new_task, best_task) < 0:
                best_insert_loc, best_task = insert_loc, new_task

        if not dry_run: # if not a dry run, then insert the task into the schedule
            schedule[best_task.node].insert(best_insert_loc, best_task)
            task_graph.nodes[task]['scheduled_task'] = best_task
        return best_task
    
    def serialize(self) -> Dict[str, Any]:
        return {
            "name": "GreedyInsert",
            "append_only": self.append_only,
            "compare": self.compare,
            "critical_path": self.critical_path
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "GreedyInsert":
        return cls(
            append_only=data["append_only"],
            compare=data["compare"],
            critical_path=data["critical_path"],
        )

class ParametricKDepthScheduler(Scheduler):
    def __init__(self,
                 scheduler: ParametricScheduler,
                 k_depth: int) -> None:
            super().__init__()
            self.scheduler = scheduler
            self.k_depth = k_depth
    
    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: Optional[ScheduleType] = None) -> ScheduleType:
        """Schedule the tasks on the network.
        
        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (Optional[ScheduleType]): The current schedule.

        Returns:
            Dict[Hashable, List[Task]]: A dictionary mapping nodes to a list of tasks executed on the node.
        """
        queue = self.scheduler.initial_priority(network, task_graph)
        schedule = {node: [] for node in network.nodes} if schedule is None else deepcopy(schedule)
        scheduled_tasks: Dict[Hashable, Task] = {}
        while queue:
            task_name = queue.pop(0)
            k_depth_successors = nx.single_source_shortest_path_length(
                G=task_graph,
                source=task_name,
                cutoff=self.k_depth
            )
            sub_task_graph = task_graph.subgraph(
                set(scheduled_tasks.keys()) | set(k_depth_successors.keys()) | {task_name}
            )
            
            best_node, best_makespan = None, float('inf')
            for node in network.nodes:
                sub_schedule = deepcopy(schedule) # copy the schedule
                _network = network.copy()
                _task_graph = task_graph.copy()
                _sub_task_graph = sub_task_graph.copy()
                self.scheduler.insert_task(_network, _task_graph, sub_schedule, task_name, node) # insert the task
                sub_schedule = self.scheduler.schedule(_network, _sub_task_graph, sub_schedule)
                sub_schedule_makespan = max(
                    task.end for tasks in sub_schedule.values() for task in tasks
                )
                if sub_schedule_makespan < best_makespan:
                    best_node = node
                    best_makespan = sub_schedule_makespan

            task = self.scheduler.insert_task(network, task_graph, schedule, task_name, best_node)
            scheduled_tasks[task.name] = task
            
        return schedule

    def serialize(self) -> Dict[str, Any]:
        """Return a dictionary representation of the initial priority.
        
        Returns:
            Dict[str, Any]: A dictionary representation of the initial priority.
        """
        return {
            **self.scheduler.serialize(),
            "name": "ParametricKDepthScheduler",
            "k_depth": self.k_depth
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "ParametricKDepthScheduler":
        """Return a new instance of the initial priority from the serialized data.
        
        Args:
            data (Dict[str, Any]): The serialized data.

        Returns:
            ParametricKDepthScheduler: A new instance of the initial priority.
        """
        return cls(
            scheduler=ParametricScheduler.deserialize(data),
            k_depth=data["k_depth"]
        )

class ParametricSufferageScheduler(ParametricScheduler):
    def __init__(self, scheduler: ParametricScheduler, top_n: int = 2):
        super().__init__(
            initial_priority=scheduler.initial_priority,
            insert_task=scheduler.insert_task
        )
        self.scheduler = scheduler
        self.top_n = top_n

    def serialize(self) -> Dict[str, Any]:
        return {
            **self.scheduler.serialize(),
            "name": "ParametricSufferageScheduler",
            "sufferage_top_n": self.top_n
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "ParametricSufferageScheduler":
        return cls(top_n=data["top_n"])
    
    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: Optional[ScheduleType] = None) -> ScheduleType:
        """Schedule the tasks on the network.
        
        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (Optional[ScheduleType]): The current schedule.

        Returns:
            Dict[Hashable, List[Task]]: A dictionary mapping nodes to a list of tasks executed on the node.
        """
        queue = self.initial_priority(network, task_graph)
        schedule = {node: [] for node in network.nodes} if schedule is None else deepcopy(schedule)
        scheduled_tasks: Dict[Hashable, Task] = {}
        while queue:
            ready_tasks = [
                task for task in queue
                if all(pred in scheduled_tasks for pred in task_graph.predecessors(task))
            ]
            max_sufferage_task, max_sufferage = None, -np.inf
            for task in ready_tasks[:self.top_n]:
                best_task = self.insert_task(network, task_graph, schedule, task, dry_run=True)
                node_weight = network.nodes[best_task.node]['weight']
                try:
                    network.nodes[best_task.node]['weight'] = 1e-9
                    second_best_task = self.insert_task(network, task_graph, schedule, task, dry_run=True)
                finally:
                    network.nodes[best_task.node]['weight'] = node_weight
                sufferage = self.insert_task._compare(network, task_graph, schedule, second_best_task, best_task)
                if sufferage > max_sufferage:
                    max_sufferage_task, max_sufferage = best_task, sufferage
            new_task = self.insert_task(network, task_graph, schedule, max_sufferage_task.name, node=max_sufferage_task.node)
            scheduled_tasks[new_task.name] = new_task
            queue.remove(new_task.name)

        return schedule


insert_funcs: Dict[str, GreedyInsert] = {}
for compare_func_name in GREEDY_INSERT_COMPARE_FUNCS.keys():
    insert_funcs[f"{compare_func_name}_Append"] = GreedyInsert(append_only=True, compare=compare_func_name, critical_path=False)
    insert_funcs[f"{compare_func_name}_Insert"] = GreedyInsert(append_only=False, compare=compare_func_name, critical_path=False)
    insert_funcs[f"{compare_func_name}_Append_CP"] = GreedyInsert(append_only=True, compare=compare_func_name, critical_path=True)
    insert_funcs[f"{compare_func_name}_Insert_CP"] = GreedyInsert(append_only=False, compare=compare_func_name, critical_path=True)

initial_priority_funcs = {
    "UpwardRanking": UpwardRanking(),
    "CPoPRanking": CPoPRanking(),
    "ArbitraryTopological": ArbitraryTopological()
}

schedulers: Dict[str, ParametricScheduler] = {}
for name, insert_func in insert_funcs.items():
    for intial_priority_name, initial_priority_func in initial_priority_funcs.items():
        reg_scheduler = ParametricScheduler(
            initial_priority=initial_priority_func,
            insert_task=insert_func
        )
        reg_scheduler.name = f"{name}_{intial_priority_name}"
        schedulers[reg_scheduler.name] = reg_scheduler

        sufferage_scheduler = ParametricSufferageScheduler(
            scheduler=reg_scheduler,
            top_n=2
        )
        sufferage_scheduler.name = f"{reg_scheduler.name}_Sufferage"
        schedulers[sufferage_scheduler.name] = sufferage_scheduler
