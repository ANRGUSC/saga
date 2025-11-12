"""Extract individual branch schedules from overlapping conditional schedule."""
from copy import deepcopy
from typing import Dict, Hashable, List, Optional, Tuple
import networkx as nx

from overlapping_scheduler import OverlappingTask


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
            
            # Check if any children are conditional
            is_conditional = any(
                task_graph.edges[current, child].get('conditional', False) 
                for child in children
            )
            
            if is_conditional:
                # Multiple conditional branches - recursively explore each one
                all_branches = []
                for child in children:
                    # Create a descriptive name for this choice
                    child_name = str(child)
                    
                    # Recursively explore this branch
                    branches_from_child = _explore_branches(
                        queue=[child] + queue.copy(),
                        visited=visited.copy(),
                        path_names=path_names + [child_name]
                    )
                    all_branches.extend(branches_from_child)
                
                return all_branches
            else:
                # Non-conditional children - add to queue
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


def extract_branch_schedule(
    original_schedule: Dict[Hashable, List[OverlappingTask]],
    branch_tasks: List[Hashable],
    conditional_group_to_keep: int = -1
) -> Dict[Hashable, List[OverlappingTask]]:
    """Extract a single branch schedule by removing tasks not in the branch.
    
    Args:
        original_schedule: The schedule with overlapping conditional tasks
        branch_tasks: List of task names that should be kept in this branch
        conditional_group_to_keep: If >= 0, only keep tasks from this conditional group
        
    Returns:
        New schedule with only the tasks in this branch
    """
    branch_schedule = {}
    
    #loop through each node on the original schedule, node = "node1", "node2", etc, tasks = [A:0-2, B:2-5, C:2-6, D:7-9] (list of OverlappingTask objects)
    for node, tasks in original_schedule.items():
        branch_schedule[node] = []
        
        for task in tasks:
            # Keep the task if:
            # 1. It's in the branch_tasks list
            # 2. AND either it's not conditional, or it's in the group we're keeping
            should_keep = task.name in branch_tasks
            
            if should_keep and task.is_conditional:
                # Additional check: if we specified a group to keep, only keep tasks from that group
                if conditional_group_to_keep >= 0:
                    should_keep = (task.conditional_group == conditional_group_to_keep)
            
            if should_keep:
                # Create a copy of the task
                new_task = OverlappingTask(
                    task.node,
                    task.name,
                    task.start,
                    task.end,
                    conditional_group=task.conditional_group,
                    is_conditional=task.is_conditional
                )
                branch_schedule[node].append(new_task)
    
    return branch_schedule


def extract_branch_schedule_multi_group(
    original_schedule: Dict[Hashable, List[OverlappingTask]],
    branch_tasks: List[Hashable],
    conditional_groups_to_keep: set
) -> Dict[Hashable, List[OverlappingTask]]:
    """Extract a single branch schedule handling multiple conditional groups.
    
    This is an enhanced version that works with branches that have multiple
    conditional branching points (e.g., A->B/C, then B->D/E).
    
    Args:
        original_schedule: The schedule with overlapping conditional tasks
        branch_tasks: List of task names that should be kept in this branch
        conditional_groups_to_keep: Set of conditional group IDs to keep
        
    Returns:
        New schedule with only the tasks in this branch
        
    Example:
        For path A->B->D (where B/C are group 0, D/E are group 1):
        branch_tasks = ["A", "B", "D"]
        conditional_groups_to_keep = {0, 1}
        
        This will:
        - Keep A (non-conditional, in branch)
        - Keep B (group 0, in branch, in groups_to_keep)
        - Remove C (group 0, not in branch)
        - Keep D (group 1, in branch, in groups_to_keep)
        - Remove E (group 1, not in branch)
    """
    branch_schedule = {}
    
    for node, tasks in original_schedule.items():
        branch_schedule[node] = []
        
        for task in tasks:
            # Keep the task if:
            # 1. It's in the branch_tasks list
            should_keep = task.name in branch_tasks
            
            # 2. AND if it's conditional, it must be in one of the groups we're keeping
            if should_keep and task.is_conditional:
                should_keep = task.conditional_group in conditional_groups_to_keep
            
            if should_keep:
                # Create a copy of the task
                new_task = OverlappingTask(
                    task.node,
                    task.name,
                    task.start,
                    task.end,
                    conditional_group=task.conditional_group,
                    is_conditional=task.is_conditional
                )
                branch_schedule[node].append(new_task)
    
    return branch_schedule


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


def print_branch_schedules(branch_schedules: Dict[str, Dict[Hashable, List[OverlappingTask]]]):
    """Print all branch schedules in a readable format."""
    print("\n" + "=" * 60)
    print("INDIVIDUAL BRANCH SCHEDULES")
    print("=" * 60)
    
    for branch_name, schedule in branch_schedules.items():
        print(f"\n{branch_name}:")
        print("-" * 40)
        
        for node in sorted(schedule.keys()):
            if schedule[node]:  # Only print nodes with tasks
                print(f"  Node {node}:")
                for task in schedule[node]:
                    print(f"    {task.name}: [{task.start:.2f}, {task.end:.2f}]")
        
        # Calculate makespan
        if any(schedule.values()):
            makespan = max(task.end for tasks in schedule.values() for task in tasks if tasks)
            print(f"  Makespan: {makespan:.2f}")
        else:
            print("  Makespan: 0.00")
    
    print("=" * 60)


def verify_branch_consistency(
    branch_schedules: Dict[str, Dict[Hashable, List[OverlappingTask]]],
    task_graph: nx.DiGraph
):
    """Verify that non-conditional tasks are scheduled consistently across all branches.
    
    This checks that tasks appearing in multiple branches are scheduled at the same
    time and on the same node in all branches.
    
    Args:
        branch_schedules: Dict of branch schedules
        task_graph: The task graph
    """
    print("\n" + "=" * 60)
    print("VERIFYING BRANCH CONSISTENCY")
    print("=" * 60)
    
    # Collect task schedules from each branch
    task_info = {}  # task_name -> [(branch_name, node, start, end)]
    
    for branch_name, schedule in branch_schedules.items():
        for node, tasks in schedule.items():
            for task in tasks:
                if task.name not in task_info:
                    task_info[task.name] = []
                task_info[task.name].append((branch_name, node, task.start, task.end, task.is_conditional))
    
    # Check consistency
    all_consistent = True
    
    for task_name, occurrences in task_info.items():
        if len(occurrences) <= 1:
            continue  # Task only appears in one branch
        
        # Check if this is a conditional task
        is_conditional = occurrences[0][4]
        
        if is_conditional:
            # Conditional tasks should NOT appear in multiple branches
            print(f"  ⚠ Task {task_name} (conditional) appears in {len(occurrences)} branches - THIS IS EXPECTED")
        else:
            # Non-conditional tasks should be consistent across branches
            first_occurrence = occurrences[0]
            first_node, first_start, first_end = first_occurrence[1], first_occurrence[2], first_occurrence[3]
            
            consistent = True
            for occurrence in occurrences[1:]:
                branch, node, start, end, _ = occurrence
                if node != first_node or start != first_start or end != first_end:
                    consistent = False
                    print(f"  ✗ Task {task_name} (non-conditional) INCONSISTENT:")
                    print(f"      Branch '{first_occurrence[0]}': Node {first_node}, [{first_start:.2f}, {first_end:.2f}]")
                    print(f"      Branch '{branch}': Node {node}, [{start:.2f}, {end:.2f}]")
                    all_consistent = False
                    break
            
            if consistent:
                print(f"  ✓ Task {task_name} (non-conditional) consistent across {len(occurrences)} branches: "
                      f"Node {first_node}, [{first_start:.2f}, {first_end:.2f}]")
    
    if all_consistent:
        print("\n✓ ALL NON-CONDITIONAL TASKS ARE CONSISTENTLY SCHEDULED!")
    else:
        print("\n✗ INCONSISTENCIES DETECTED - this may indicate a scheduling problem")
    
    print("=" * 60)

