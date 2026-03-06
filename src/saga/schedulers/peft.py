import numpy as np
from typing import Dict, Optional, List, Tuple
from saga import Schedule, Scheduler, ScheduledTask, TaskGraph, Network, NetworkNode


def OCT_table(task_graph: TaskGraph, network: Network) -> Dict[str, Dict[str, float]]:
    """Computes the Optimistic Cost Table (OCT) for each task and processor.

    Args: 
        task_graph (TaskGraph): The task graph.
        network (Network): The network graph.
    
    Returns:
        Dict[str, Dict[str, float]]: A mapping from each task name to each
            processor name and its optimistic future cost.
    """
    table: Dict[str, Dict[str, float]] = {}

    for task in reversed(task_graph.topological_sort()):
        if task.name not in table: table[task.name] = {}
        for processor in network.nodes:
            child_edges = task_graph.out_edges(task.name)
            if not child_edges: 
                table[task.name][processor.name] = 0.0  
                continue
            child_path_costs: list[float] = []

            for edge in child_edges:
                child_name = edge.target
                min_child_path_cost = np.inf
                child_task = task_graph.get_task(child_name)
                for next_processor in network.nodes:
                    if processor.name == next_processor.name: pass_time = 0.0
                    else: pass_time = (
                        task_graph.get_dependency(task.name, child_name).size / 
                        network.get_edge(processor.name, next_processor.name).speed
                    )
                    next_task_time = child_task.cost / next_processor.speed
                    candidate_cost = pass_time + next_task_time + table[child_name][next_processor.name] 
                    min_child_path_cost = min(candidate_cost, min_child_path_cost)
                child_path_costs.append(min_child_path_cost)            
            table[task.name][processor.name] = max(child_path_costs)
    return table

def peft_rank_sort(task_graph: TaskGraph, oct_table: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, float], List[str]]:
    """Computes and sorts the average OCT for each task (as defined in the PEFT paper).

    Args:
        task_graph (TaskGraph): The task graph.
        oct_table (Dict[str, Dict[str, float]]): The Optimistic Cost Table.

    Returns:
        Tuple[Dict[str, float], List[str]]: A mapping from task names to their average OCT values, 
            and the tasks sorted by those values.
    """
    rank_oct = {
        task: sum(proc_map.values()) / len(proc_map) if proc_map else 0.0 for task, proc_map in oct_table.items()
    }
    rev_topo_sort = {
        node.name: i for i, node in enumerate(reversed(task_graph.topological_sort()))
    }
    order = sorted(
        rank_oct.keys(), key=lambda x: (rank_oct[x], rev_topo_sort.get(x, -1)), reverse=True
    )
    return rank_oct, order

def select_best_processor(
        task_name: str, 
        task_graph: TaskGraph, 
        network: Network, 
        oct_table: Dict[str, Dict[str, float]],
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0
    ) -> Tuple[NetworkNode, float, float]:
    """Selects the processor that minimizes the optimistic earliest finish time for a task.

    Args: 
        task_name (str): The name of the task to schedule.
        task_graph (TaskGraph): The task graph.
        network (Network): The network graph.
        oct_table (Dict[str, Dict[str, float]]): The Optimistic Cost Table.
        schedule (Schedule): The schedule. Defaults to None.
        min_start_time (float): The minimum start time. Defaults to 0.0.

    Returns:
        Tuple[NetworkNode, float, float]: The best processor, start time, and end time for the task.
    """
    best_start_time = 0.0
    best_end_time = 0.0
    best_rank = np.inf
    best_processor = None
    for processor in network.nodes: 
        if schedule is not None:
            start_time = schedule.get_earliest_start_time(
                task=task_name, node=processor.name, append_only=False
            )
            start_time = max(start_time, min_start_time)
            earliest_finish_time = start_time + (
                task_graph.get_task(task_name).cost / processor.speed
            )
            predicted_cost = oct_table[task_name][processor.name]
            optimistic_eft = predicted_cost + earliest_finish_time
            if optimistic_eft < best_rank:
                best_start_time = start_time
                best_end_time = earliest_finish_time
                best_rank = optimistic_eft
                best_processor = processor
    if best_processor == None: raise RuntimeError("No processor")
    return (best_processor, best_start_time, best_end_time)


class PEFTScheduler(Scheduler):
    """
    Schedules tasks using the Predict Earliest Finish Time (PEFT) algorithm.
    
    Source: https://ieeexplore.ieee.org/document/6471969 
    """
    def schedule(
            self, 
            network: Network, 
            task_graph: TaskGraph, 
            schedule: Optional[Schedule] = None, 
            min_start_time: float = 0.0
        ) -> Schedule:
        """Schedules the tasks on the network.

        Args: 
            network (Network): The network graph.
            task_graph (TaskGraph): The task graph.
            schedule (Schedule): The schedule. Defaults to None.
            min_start_time (float): The minimum start time. Defaults to 0.0.
        
        Returns: 
            Schedule: The schedule.
        """
        oct_table = OCT_table(task_graph, network)
        rank_oct, peft_tasks = peft_rank_sort(task_graph, oct_table)
        schedule = Schedule(task_graph, network)
        ready_list: list[str] = []
        peft_order_idx = {task_name: i for i, task_name in enumerate(peft_tasks)}

        for task in peft_tasks:
            if len(task_graph.in_edges(task)) == 0: 
                ready_list.append(task)

        while ready_list:
            task_name = max(ready_list, key=lambda x: (rank_oct[x], -peft_order_idx[x]))         
            best_processor, best_start_time, best_end_time = select_best_processor(
                task_name, task_graph, network, oct_table, schedule, min_start_time
            )

            new_task = ScheduledTask(
                node=best_processor.name,
                name=task_name,
                start=best_start_time,
                end=best_end_time
            )
            schedule.add_task(new_task)
            ready_list.remove(task_name) 

            for child_edges in task_graph.out_edges(task_name):
                child_name = child_edges.target
                parent_edges = task_graph.in_edges(child_name)
                if all(
                    schedule.is_scheduled(edge.source) for edge in parent_edges) and (
                    not schedule.is_scheduled(child_name) and (
                        child_name not in ready_list
                )):
                    ready_list.append(child_name)
        return schedule