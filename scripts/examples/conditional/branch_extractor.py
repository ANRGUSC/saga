"""Extract individual branch schedules from overlapping conditional schedule."""
from copy import deepcopy
from typing import Dict, Hashable, List, Optional, Tuple
import networkx as nx

from overlapping_scheduler import OverlappingTask
from saga.schedulers.heft import HeftScheduler
from saga.scheduler import Task


def identify_branches(task_graph: nx.DiGraph) -> List[Tuple[str, List[Hashable]]]:
    """Identify all possible execution branches in the conditional task graph.
    
    Uses a recursive BFS-like approach similar to bfs.py to handle multiple branching points.
    
    Args:
        task_graph: The conditional task graph
        
    Returns:
        List of (branch_name, tasks_in_branch) tuples
        
    Example:
        For A -> B/C -> D where both B and C lead to D:
        Returns: [("Path: B", ["A", "B", "D"]), ("Path: C", ["A", "C", "D"])]
        
        For A -> B/C, B -> D/E (multiple branches):
        Returns: [("Path: B-D", ["A", "B", "D"]), 
                  ("Path: B-E", ["A", "B", "E"]),
                  ("Path: C", ["A", "C"])]
    """
    def _explore_branches(queue: Optional[List] = None, 
                          visited: Optional[set] = None,
                          path_names: Optional[List[str]] = None) -> List[Tuple[List[str], set]]:
        """Recursively explore all possible branches.
        
        Args:
            queue: Nodes to process (BFS queue)
            visited: Nodes already visited in this path
            path_names: Names of conditional choices made (for branch naming)
            
        Returns:
            List of (path_name_list, visited_nodes_set) tuples
        """
        # Initialize on first call
        if queue is None:
            queue = [node for node in task_graph.nodes if task_graph.in_degree(node) == 0]
        if visited is None:
            visited = set()
        if path_names is None:
            path_names = []
        
        # BFS traversal
        while queue:
            current = queue.pop(0)
            visited.add(current)
            
            # Get all children of current node
            children = list(task_graph.successors(current))
            
            # Separate conditional from non-conditional children
            conditional_children = [
                child for child in children 
                if task_graph.edges[current, child].get('conditional', False)
            ]
            non_conditional_children = [
                child for child in children 
                if not task_graph.edges[current, child].get('conditional', False)
            ]
            
            if conditional_children:
                # Multiple conditional branches - recursively explore each one
                all_branches = []
                for child in conditional_children:
                    # Create a descriptive name for this choice
                    child_name = str(child)
                    
                    # Build queue for this branch: conditional child + non-conditional children + rest of queue
                    new_queue = [child] + non_conditional_children + queue.copy()
                    
                    # Recursively explore this branch
                    branches_from_child = _explore_branches(
                        queue=new_queue,
                        visited=visited.copy(),
                        path_names=path_names + [child_name]
                    )
                    all_branches.extend(branches_from_child)
                
                return all_branches
            else:
                # No conditional children - add all to queue (continue current path)
                for child in children:
                    if child not in visited and child not in queue:
                        queue.append(child)
        
        # Reached the end of this path
        return [(path_names, visited)]
    
    # Explore all branches
    raw_branches = _explore_branches()
    
    # Convert to the expected format
    branches = []
    for path_names, visited_nodes in raw_branches:
        if path_names:
            # Create branch name from the path choices
            branch_name = f"Path: {'-'.join(path_names)}"
        else:
            # No conditional choices (all non-conditional)
            branch_name = "No conditionals"
        
        branches.append((branch_name, sorted(visited_nodes)))
    
    return branches


def extract_all_branches(
    original_schedule: Dict[Hashable, List[OverlappingTask]],
    task_graph: nx.DiGraph
) -> Dict[str, Dict[Hashable, List[OverlappingTask]]]:
    """Extract all possible branch schedules from the overlapping schedule.
    
    Simple approach: identify_branches() already gives us the correct task names
    for each branch, so we just filter the schedule to keep only those tasks.
    
    Args:
        original_schedule: The schedule with overlapping conditional tasks
        task_graph: The conditional task graph
        
    Returns:
        Dict mapping branch names to their individual schedules
    """
    branches = identify_branches(task_graph)
    branch_schedules = {}
    
    # For each branch, keep only the tasks in that branch
    for branch_name, branch_tasks in branches:
        branch_schedule = {}
        
        for node, tasks in original_schedule.items():
            branch_schedule[node] = []
            
            for task in tasks:
                # Simple: just check if task name is in this branch
                if task.name in branch_tasks:
                    # Copy the task
                    new_task = OverlappingTask(
                        task.node,
                        task.name,
                        task.start,
                        task.end,
                        conditional_group=task.conditional_group,
                        is_conditional=task.is_conditional
                    )
                    branch_schedule[node].append(new_task)
        
        branch_schedules[branch_name] = branch_schedule
    
    return branch_schedules


def recalculate_branch_times(
    branch_schedule: Dict[Hashable, List[OverlappingTask]],
    task_graph: nx.DiGraph,
    runtimes: Dict[Hashable, Dict[Hashable, float]],
    commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]
) -> Dict[Hashable, List[OverlappingTask]]:
    """Recalculate task times to remove gaps caused by other conditional branches.
    
    Keeps same node assignments, just recalculates start/end times based on 
    actual dependencies in this specific branch.
    """
    # flattens the branch_schedule dict,list into -> lookup dict allows to quickly see what node task is on
    task_info = {task.name: (node, task) 
                 for node, tasks in branch_schedule.items() 
                 for task in tasks}
    """
        task_info = {
        'A': (1, OverlappingTask(node=1, name='A', start=0.0, end=2.0, ...)),
    """

    # Track calculated end times
    end_times = {}  # task_name -> end_time
    # Fresh container
    new_schedule = {node: [] for node in branch_schedule.keys()}
    """
        new_schedule = {
            1: [],  # Will hold recalculated tasks for Node 1
            2: []   # Will hold recalculated tasks for Node 2
        }
    """
    
    # Process in topological order (only tasks in this branch)
    for task_name in nx.topological_sort(task_graph):
        if task_name not in task_info:
            continue  # Skip tasks not in this branch
        
        node, original = task_info[task_name]
        
        # Earliest start = max of (predecessor end + communication cost)
        predecessors = [task for task in task_graph.predecessors(task_name) if task in task_info]
        if predecessors:
            start = max(
                end_times[task] + commtimes.get((task_info[task][0], node), {}).get((task, task_name), 0.0)
                for task in predecessors
            )
        else:
            start = 0.0
        
        end = start + runtimes[node][task_name]
        end_times[task_name] = end
        
        new_schedule[node].append(OverlappingTask(
            node, task_name, start, end,
            original.conditional_group, original.is_conditional
        ))
    
    # Sort by start time
    for node in new_schedule:
        new_schedule[node].sort(key=lambda t: t.start)
    
    return new_schedule


def extract_all_branches_with_recalculation(
    original_schedule: Dict[Hashable, List[OverlappingTask]],
    task_graph: nx.DiGraph,
    runtimes: Dict[Hashable, Dict[Hashable, float]],
    commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]
) -> Dict[str, Dict[Hashable, List[OverlappingTask]]]:
    """Extract all branches AND recalculate their times to remove gaps.
    
    Args:
        original_schedule: The schedule with overlapping conditional tasks
        task_graph: The conditional task graph
        runtimes: Dict mapping nodes to task runtimes (from scheduler.get_runtimes())
        commtimes: Dict mapping node pairs to communication times (from scheduler.get_runtimes())
        
    Returns:
        Dict mapping branch names to their individual schedules with realistic timing
    """
    # First extract branches (keeps original timing with gaps)
    branch_schedules = extract_all_branches(original_schedule, task_graph)
    
    # Then recalculate times for each branch
    recalculated_schedules = {}
    for branch_name, branch_schedule in branch_schedules.items():
        recalculated_schedules[branch_name] = recalculate_branch_times(
            branch_schedule, task_graph, runtimes, commtimes
        )
    
    return recalculated_schedules


def generate_heft_comparison_schedules(
    task_graph: nx.DiGraph,
    network: nx.DiGraph
) -> Dict[str, Dict[Hashable, List[Task]]]:
    """Generate standard HEFT schedules for each branch to compare against overlapping approach.
    
    For each branch, creates a static DAG containing only the tasks in that branch,
    then runs standard HEFT scheduling without conditional overlapping logic.
    
    Args:
        task_graph: The conditional task graph
        network: The heterogeneous computing network
        
    Returns:
        Dict mapping branch names to their HEFT schedules (as standard Task objects)
    """
    branches = identify_branches(task_graph)
    heft_schedules = {}
    
    for branch_name, branch_tasks in branches:
        # Create subgraph containing only tasks in this branch
        branch_subgraph = task_graph.subgraph(branch_tasks).copy()
        
        # Remove conditional edge attributes (make it a static DAG)
        for u, v in branch_subgraph.edges():
            if 'conditional' in branch_subgraph.edges[u, v]:
                del branch_subgraph.edges[u, v]['conditional']
            if 'probability' in branch_subgraph.edges[u, v]:
                del branch_subgraph.edges[u, v]['probability']
        
        # Run standard HEFT on this branch
        scheduler = HeftScheduler()
        heft_schedule = scheduler.schedule(network, branch_subgraph)
        
        heft_schedules[branch_name] = heft_schedule
    
    return heft_schedules


