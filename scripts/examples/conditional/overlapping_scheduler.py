"""New Conditional HEFT Scheduler that allows conditional tasks to overlap."""
import sys
import pathlib
from copy import deepcopy
from typing import Dict, Hashable, List, Optional, Tuple
import networkx as nx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.parent / "src"))

from saga.scheduler import Task
from saga.utils.tools import get_insert_loc


def identify_conditional_groups(task_graph: nx.DiGraph) -> Dict[Hashable, int]:
    """
    Goes through each node, is has conditional children -> puts them into group 
    else -> puts all non conditionals in group
    
    Args:
        task_graph: The task graph with 'conditional' and 'probability' edge attributes
        
    Returns:
        Dict mapping task names to group IDs. Tasks with the same group ID are conditional
        alternatives and should be scheduled as if they overlap. Group ID -1 means non-conditional.

        e.g.
        {
            "A": -1,  # Non-conditional
            "B": 0,   # Conditional group 0 
            "C": 0,   # Conditional group 0 
            "D": 1,   # Conditional group 1 
            "E": 1,   # Conditional group 1 
            "F": 1,   # Conditional group 1 
            "G": -1,  # Non-conditional
            "H": -1,  # Non-conditional
            "I": -1   # Non-conditional
        }
    """
    groups = {}
    group_id = 0
    
    for node in task_graph.nodes:
        # Check all children of this node
        children = list(task_graph.successors(node))
        
        # Check if this node has conditional branches
        conditional_children = []
        for child in children:
            edge_data = task_graph.edges[node, child]
            if edge_data.get('conditional', False):
                conditional_children.append(child)
        
        # If there are conditional children, they form a group
        if len(conditional_children) > 1:
            for child in conditional_children:
                groups[child] = group_id
            group_id += 1
    
    # Mark non-conditional tasks
    for node in task_graph.nodes:
        if node not in groups:
            groups[node] = -1
    
    return groups


def compute_conditional_rank(network: nx.Graph, 
                              task_graph: nx.DiGraph,
                              conditional_groups: Dict[Hashable, int]) -> Dict[Hashable, float]:
    """Compute upward rank with conditional tasks getting the same rank.
    
    Tasks in the same conditional group get the same rank (maximum of their individual ranks)
    so they will be scheduled together.
    
    Args:
        network: The network graph
        task_graph: The task graph
        conditional_groups: Mapping of tasks to their conditional group IDs
        
    Returns:
        Dict mapping task names to their ranks
    """
    ranks = {}
    
    #standard upward rank
    topological_order = list(nx.topological_sort(task_graph))
    for node in topological_order[::-1]:
        # Average computation time across all processors
        avg_comp_time = np.mean([
            task_graph.nodes[node]['weight'] / network.nodes[proc]['weight']
            for proc in network.nodes
        ])
        #Maximum of (successor rank + communication time)
        max_successor_cost = 0 if task_graph.out_degree(node) <= 0 else max(
            [
                ranks[neighbor] + np.mean([
                    task_graph.edges[node, neighbor]['weight'] / network.edges[src, dst]['weight']
                    for src, dst in network.edges
                ])
                for neighbor in task_graph.successors(node)
            ]
        )
        ranks[node] = avg_comp_time + max_successor_cost
    
    #New code for upward rank with conditional tasks getting the same rank
    #Find all conditional groups
    group_tasks = {}
    for task, group_id in conditional_groups.items():
        if group_id >= 0:  #Skip non-conditional tasks
            if group_id not in group_tasks:
                group_tasks[group_id] = []
            group_tasks[group_id].append(task)
    
    #Set all tasks in each group to the maximum rank in that group
    for group_id, tasks in group_tasks.items():
        max_rank = None
        for task in tasks:
            rank_value = ranks[task]
            if max_rank is None or rank_value > max_rank:
                max_rank = rank_value
        for task in tasks:
            ranks[task] = max_rank
    
    return ranks


def rank_sort_with_conditional_grouping(network: nx.Graph,
                                        task_graph: nx.DiGraph,
                                        conditional_groups: Dict[Hashable, int]) -> List[Hashable]:
    """Sort tasks by rank, with conditional tasks grouped together.
    
    Uses the same approach as heft_rank_sort: adds rank and topological order.
    Conditional tasks have the same rank, so topological order acts as tie-breaker.
    
    Example:
        combined_rank = {
            "A": 15.0 + 1,
            "B": 10.0 + 5,
            "C": 10.0 + 4,
            ...
        }
    """

    '''
        #new code: Create combined ranking -------------- new method instead of combining rank and topological it creates both and only uses 
        combined_rank = {}
        for node in ranks:
            #create tuple with (descending),(topological order)
            combined_rank[node] = (ranks[node], topological_sort[node])
        
        # Sort by combined rank
        sorted_tasks = sorted(list(combined_rank.keys()), 
                            key=lambda x: combined_rank[x], 
                            reverse=True)
        
        return sorted_tasks
    '''
    ranks = compute_conditional_rank(network, task_graph, conditional_groups)
    topological_sort = {node: i for i, node in enumerate(reversed(list(nx.topological_sort(task_graph))))}
    
    # Combine rank and topological sort (same approach as heft_rank_sort)
    combined_rank = {node: (ranks[node] + topological_sort[node]) for node in ranks}
    
    # Sort by combined rank
    return sorted(list(combined_rank.keys()), key=combined_rank.get, reverse=True)



class OverlappingTask(Task):
    """inherits Task class, and adds conditional variables to it"""
    
    def __init__(self, node, name, start, end, conditional_group=-1, is_conditional=False):
        super().__init__(node, name, start, end)
        self.conditional_group = conditional_group
        self.is_conditional = is_conditional


class OverlappingConditionalScheduler:
    """Scheduler that allows conditional tasks to overlap in the schedule."""
    
    @staticmethod
    def get_runtimes(network: nx.Graph, 
                     task_graph: nx.DiGraph) -> Tuple[
                         Dict[Hashable, Dict[Hashable, float]],
                         Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]]:
        """Get the expected runtimes of all tasks on all nodes. same as heft.py"""
        runtimes = {}
        for node in network.nodes:
            runtimes[node] = {}
            speed = network.nodes[node]["weight"]
            for task in task_graph.nodes:
                cost = task_graph.nodes[task]["weight"]
                runtimes[node][task] = cost / speed
        
        commtimes = {}
        for src, dst in network.edges:
            if src == dst:  # Skip self-loops for communication
                continue
            commtimes[src, dst] = {}
            commtimes[dst, src] = {}
            speed = network.edges[src, dst]["weight"]
            for src_task, dst_task in task_graph.edges:
                cost = task_graph.edges[src_task, dst_task]["weight"]
                commtimes[src, dst][src_task, dst_task] = cost / speed
                commtimes[dst, src][src_task, dst_task] = cost / speed
        
        # Add zero-cost communication for same-node tasks
        for node in network.nodes:
            commtimes[node, node] = {}
            for src_task, dst_task in task_graph.edges:
                commtimes[node, node][src_task, dst_task] = 0.0
        
        return runtimes, commtimes
    
    def schedule(self, 
                 network: nx.Graph, 
                 task_graph: nx.DiGraph,
                 schedule: Optional[Dict[Hashable, List[OverlappingTask]]] = None,
                 min_start_time: float = 0.0) -> Dict[Hashable, List[OverlappingTask]]:
        """Schedule tasks allowing conditional alternatives to overlap.
        
        Args:
            network: The network graph
            task_graph: The task graph with 'conditional' edge attributes
            schedule: Optional existing schedule to extend (for online scheduling)
            min_start_time: Minimum start time for new tasks (for online scheduling)
            
        Returns:
            Schedule dict mapping nodes to lists of OverlappingTask objects
        """
        # Identify conditional groups
        conditional_groups = identify_conditional_groups(task_graph)       
        # Create ranks and sort ranks
        schedule_order = rank_sort_with_conditional_grouping(network, task_graph, conditional_groups)
        # Get runtimes and communication times
        runtimes, commtimes = self.get_runtimes(network, task_graph)
        
        # Delegate to _schedule for actual scheduling work
        return self._schedule(network, task_graph, runtimes, commtimes, 
                            schedule_order, conditional_groups, schedule, min_start_time)
    
    def _schedule(self,
                  network: nx.Graph,
                  task_graph: nx.DiGraph,
                  runtimes: Dict[Hashable, Dict[Hashable, float]],
                  commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
                  schedule_order: List[Hashable],
                  conditional_groups: Dict[Hashable, int],
                  schedule: Optional[Dict[Hashable, List[OverlappingTask]]] = None,
                  min_start_time: float = 0.0) -> Dict[Hashable, List[OverlappingTask]]:
        """Schedule the tasks on the network with conditional task overlapping.
        
        Args:
            network: The network graph
            task_graph: The task graph
            runtimes: Dictionary mapping nodes to task runtimes
            commtimes: Dictionary mapping edges to task communication times
            schedule_order: The order in which to schedule tasks
            conditional_groups: Mapping of tasks to their conditional group IDs
            schedule: Optional existing schedule to extend
            min_start_time: Minimum start time for new tasks
            
        Returns:
            Schedule dict mapping nodes to lists of OverlappingTask objects
        """
        # Initialize schedules
        if schedule is None:
            comp_schedule: Dict[Hashable, List[OverlappingTask]] = {node: [] for node in network.nodes}
            task_schedule: Dict[Hashable, OverlappingTask] = {}
        else:
            comp_schedule = deepcopy(schedule)
            task_schedule = {task.name: task for node in schedule for task in schedule[node]}
        
        #Schedule tasks in order, using index to skip past groups
        i = 0
        while i < len(schedule_order):
            task_name = schedule_order[i]
            
            if task_name in task_schedule:
                i += 1
                continue
            
            #Check if this task is part of a conditional group
            group_id = conditional_groups[task_name]
            
            if group_id >= 0:
                # This is a conditional task find all tasks in this group
                group_tasks = [t for t in schedule_order[i:] 
                              if conditional_groups[t] == group_id]
                                
                #Schedule all conditional tasks in this group as if they overlap
                for cond_task in group_tasks:
                    self._schedule_single_task(
                        cond_task, network, task_graph, runtimes, commtimes,
                        comp_schedule, task_schedule, conditional_groups,
                        allow_overlap=True, current_group=group_id,
                        min_start_time=min_start_time
                    )
                
                #Skip past all tasks in this group
                i += len(group_tasks)
            else:
                #Non-conditional task - schedule normally
                self._schedule_single_task(
                    task_name, network, task_graph, runtimes, commtimes,
                    comp_schedule, task_schedule, conditional_groups,
                    allow_overlap=False, current_group=-1,
                    min_start_time=min_start_time
                )
                i += 1
        
        return comp_schedule
    
    def _schedule_single_task(self,
                             task_name: Hashable,
                             network: nx.Graph,
                             task_graph: nx.DiGraph,
                             runtimes: Dict[Hashable, Dict[Hashable, float]],
                             commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
                             comp_schedule: Dict[Hashable, List[OverlappingTask]],
                             task_schedule: Dict[Hashable, OverlappingTask],
                             conditional_groups: Dict[Hashable, int],
                             allow_overlap: bool,
                             current_group: int,
                             min_start_time: float = 0.0):
        """Schedule a single task on the best node.
        
        If allow_overlap=True, this task can be scheduled at the same time as other tasks
        in the same conditional group.
        
        Args:
            min_start_time: Minimum start time for this task (for online scheduling)
        """
        min_finish_time = np.inf
        best_node = None
        best_start = None
        #used mainly from Insert_loc function it stores ID of position in list where task should be inserted
        best_idx = None
        
        for node in network.nodes:
            # Same code as Heft, Calculate earliest start time based on predecessor completion for all nodes then pick that node
            predecessors = list(task_graph.predecessors(task_name))
            
            if not predecessors:
                max_arrival_time = min_start_time
            else:
                arrival_times = [min_start_time]
                for parent in predecessors:
                    if parent in task_schedule:
                        parent_task = task_schedule[parent]
                        comm_time = commtimes.get((parent_task.node, node), {}).get((parent, task_name), 0)
                        arrival_times.append(parent_task.end + comm_time)
                max_arrival_time = max(arrival_times)
            
            runtime = runtimes[node][task_name]
            
            # New code for overlappuing, once picked that node run get insert loc
            if allow_overlap:
                #for overlapping
                idx, start_time = self._get_insert_loc_with_overlap(
                    comp_schedule[node], max_arrival_time, runtime, current_group
                )
            else:
                #Normal scheduling - no overlap allowed
                idx, start_time = get_insert_loc(comp_schedule[node], max_arrival_time, runtime)
            
            # Same code from Heft, tracking best time
            finish_time = start_time + runtime          
            if finish_time < min_finish_time:
                min_finish_time = finish_time
                best_node = node
                best_start = start_time
                best_idx = idx
        
        # Create task and insert into schedule
        is_conditional = conditional_groups[task_name] >= 0
        task = OverlappingTask(
            best_node, task_name, best_start, min_finish_time,
            conditional_group=conditional_groups[task_name],
            is_conditional=is_conditional
        )
        
        comp_schedule[best_node].insert(best_idx, task)
        task_schedule[task_name] = task
        

    
    def _get_insert_loc_with_overlap(self,
                                     schedule: List[OverlappingTask],
                                     min_start_time: float,
                                     exec_time: float,
                                     current_group: int) -> Tuple[int, float]:
        """Find insertion location allowing overlap with tasks in the same conditional group.
        
        This function follows the same structure as get_insert_loc() from saga.utils.tools,
        but filters out tasks in the same conditional group before finding gaps.
        
        Args:
            schedule: Current schedule on this node
            earliest_start: Earliest time this task can start
            duration: Duration of the task
            current_group: Conditional group ID of the task being scheduled
        
        Returns:
            (index, start_time) where the task should be inserted
        """
        #remove all tasks in same group from schedule
        non_overlapping_tasks = [t for t in schedule 
                                 if t.conditional_group != current_group 
                                 or not t.is_conditional]
        
        #Case 1: if no tasks in schedule than, schedule in beginning
        if not non_overlapping_tasks or min_start_time + exec_time <= non_overlapping_tasks[0].start:
            #if not non_overlapping_tasks:
            return 0, min_start_time
            #else:
            #    #Fits before first non-overlapping task ------- this commented out code was for testing whether there should be order between overlapping tasks
            #    actual_idx = schedule.index(non_overlapping_tasks[0])
            #    return actual_idx, min_start_time
        
        #Case 2: Check gaps between consecutive non-overlapping tasks
        for i, (left, right) in enumerate(zip(non_overlapping_tasks, non_overlapping_tasks[1:]), start=1):
            #can start at earliest start? Note we have to fit "i" back into the schedule so thats what actual_idx is for
            if min_start_time >= left.end and min_start_time + exec_time <= right.start:
                actual_idx = schedule.index(left) + 1 #new code
                return actual_idx, min_start_time
            #wait for left to finish?
            elif min_start_time < left.end and left.end + exec_time <= right.start:
                actual_idx = schedule.index(left) + 1 #new code
                return actual_idx, left.end
        
        # Case 3: No gaps found - schedule after last non-overlapping task
        actual_idx = schedule.index(non_overlapping_tasks[-1]) + 1
        return actual_idx, max(min_start_time, non_overlapping_tasks[-1].end)


