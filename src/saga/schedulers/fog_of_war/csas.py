import pathlib
from typing import List, Optional, Dict, Tuple
import numpy as np

from saga import Schedule, Scheduler, ScheduledTask, TaskGraph, Network
from saga.schedulers.cpop import upward_rank
from saga.schedulers.fog_of_war import TaskGraphNodeFOW, TaskGraphFOW

thisdir = pathlib.Path(__file__).resolve().parent


"""
CSAS: Competative Sufferage Aware Scheduling 
# another name could be PoPS: Place or Pass Sufferage Scheduler

Operates under the NATO proposed "fog of war" scheduling. Multiple parties 
submitting jobs to a shared compute pool, with each party having access to 
different amounts of said compute pool.

Networks are modeled as clusters of high bandwidth nodes, with each cluster
connected via relatively low bandwidth. This represents regions of compute. 
For example a network may look like:


***USA view***

        USA                                     Canada
    1--(9)--2                            
    | \   / |                                  
   (9)  \  (9) ===============(1)============ 5--(9)--6
    | /   \ |                               
    3--(9)--4                                   
                                                 

***Canada view***
       USA                                     Canada
                                            3---(9)---4
                                             \       /
      1--(9)--2 ===============(1)============(9)  (9)
                                                \  /
                                                 5

A regions tasks may only run on the nodes which are visible to them.

###SCHEDULING###
Because most tasks in this problem are long-running, the primary success metric
is throughput. There is the additional challenge of being unaware of the 
exact scheduling decesions of other regions. To ensure fairness while minimizing
"stepping on toes," we propose the following:

***Shared job queue***
Schedulers like HEFT use a queue (defined by the rank of the tasks), but our problem is 
compicated by the fact that multiple regions are submitting jobs to the same queue, and 
submissions may arrive at diffenet times. Because of this, we must use a frontier queue,



Say USA submits a task graph A, and Canada submits a task graph B.
Task are thus represented as Region[rank] in a queue:

global_queue = {B[10], A[9], A[5], B[4], B[3], A[2], B[1]}

***Scheduling - Place or Pass***
When a job arrives at the front of the queue, the scheduler has two options.

Place: The scheduler can schedule that task, and any following tasks from the same region
on any visible node. Tasks can be placed on occupide nodes.
Pass: The scheduler can, based on some metric (in this implimentation sufferage), choose to 
pass on the current task(s)
"""



def allocation_tiebreaker(regions: Dict[str, Tuple[Network, TaskGraph]]) -> str:
    """
    Tiebreaker for allocating tasks to regions. The region with most donated compute gets priority.
    If there is a tie, the region with the least tasks gets priority.

    Args:
        regions (Dict[str, Tuple[Network, TaskGraph]]): The regions and their associated networks and task graphs.
    Returns:
        List[TaskGraphNode]: A list of tasks in the order of their allocation priority.
    
        ##TODO: In order to impliment this tiebreaker, we must first impliment the ability to tie network nodes to 
        regions.
    """
    pass



def _upward_rank(regions: Dict[str, Tuple[Network, TaskGraph]]) -> Dict[str, float]:
    """Computes the upward ranks of the tasks in the task graph.

    Args:
        network (Network): The network graph.
        task_graph (TaskGraph): The task graph.

    Returns:
        Dict[Hashable, float]: The upward ranks of the tasks.
    
        ##TODO: We need to update this to handle multiple regions and multiple task graphs. This involves
        computing upward ranks for each task based on the compute they have access to. Additionally, 
        topological sort is not a fair metric. A new tiebreaker based on donated compute must be established.
    """
    ranks: Dict[str, float] = {}
    tasks = [tasks+ task_list for _, (_, task_list) in regions.items()]

    topological_order = task_graph.topological_sort()
    for task in topological_order[::-1]:
        avg_comp_time = np.mean([task.cost / node.speed for node in network.nodes])
        max_comm_time = (
            0
            if task_graph.out_degree(task.name) <= 0
            else max(
                [
                    ranks[task_graph_dependency.target]
                    + np.mean(
                        [
                            task_graph_dependency.size / network_edge.speed
                            for network_edge in network.edges
                        ]
                    )
                    for task_graph_dependency in task_graph.out_edges(task.name)
                ]
            )
        )
        ranks[task.name] = float(avg_comp_time + max_comm_time)

    return ranks






def heft_rank_sort(networks: List[Tuple[Network, str]], task_graphs: List[Tuple[TaskGraph, str]]) -> List[str]:
    """Sort tasks based on their rank (as defined in the HEFT paper).

    Args:
        networks (List[Network]): The list of network graphs.
        task_graphs (List[TaskGraph]): The list of task graphs.

    Returns:
        List[str]: The sorted list of tasks.
    """
    regions: Dict[str, (Network, TaskGraph)] = {}
    for network, region in networks:
        regions[region] = (network, None)  
    for task_graph, region in task_graphs:
        regions[region] = (regions[region][0], task_graph)
    # Assuming all task graphs are on the same network
    for region, (network, task_graph) in regions.items():
        if task_graph is None:
            raise ValueError(f"No task graph provided for region {region}.")
        else:
            rank = _upward_rank(network, task_graph)


    urank = upward_rank(networks, task_graphs[0])  # Assuming all task graphs are on the same network
    topological_sort = {
        node.name: i for i, node in enumerate(reversed(task_graphs[0].topological_sort()))
    }
    rank = {node: (urank[node], topological_sort[node]) for node in urank}
    order = sorted(list(rank.keys()), key=lambda x: rank.get(x, 0.0), reverse=True)
    return order


class HeftScheduler(Scheduler):
    """Schedules tasks using the HEFT algorithm.

    Source: https://dx.doi.org/10.1109/71.993206
    """

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (Optional[Schedule], optional): An existing partial schedule to
                extend. Defaults to None.
            min_start_time (float, optional): The minimum start time. Defaults to 0.0.

        Returns:
            Schedule: The resulting schedule.

        Raises:
            ValueError: If the instance is invalid.
        """
        schedule_order = heft_rank_sort(network, task_graph)
        schedule = schedule if schedule is not None else Schedule(task_graph, network)

        for task_name in schedule_order:
            if schedule.is_scheduled(task_name):
                continue
            min_finish_time = np.inf
            best_node = next(iter(network.nodes))  # arbitrary initialization
            for node in network.nodes:
                start_time = schedule.get_earliest_start_time(
                    task=task_name,
                    node=node,
                    append_only=False,
                    current_moment=min_start_time,
                )
                runtime = (
                    task_graph.get_task(task_name).cost / network.get_node(node).speed
                )
                finish_time = start_time + runtime
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_node = node
            new_task = ScheduledTask(
                node=best_node.name,
                name=task_name,
                start=min_finish_time
                - (
                    task_graph.get_task(task_name).cost
                    / network.get_node(best_node).speed
                ),
                end=min_finish_time,
            )
            schedule.add_task(new_task)

        return schedule
