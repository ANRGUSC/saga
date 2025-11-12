"""Visualization tools for schedules with overlapping conditional tasks."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Hashable, List
import networkx as nx

from overlapping_scheduler import OverlappingTask


def draw_overlapping_gantt(schedule: Dict[Hashable, List[OverlappingTask]], 
                           task_graph: nx.DiGraph,
                           title: str = "Schedule with Overlapping Conditional Tasks",
                           save_path: str = None):
    """Draw a Gantt chart that shows overlapping conditional tasks.
    
    Conditional tasks that overlap are shown with hatching/transparency to indicate
    that only one will execute at runtime.
    
    Args:
        schedule: The schedule with OverlappingTask objects
        task_graph: The task graph (for reference)
        title: Title for the chart
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib axes object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get all nodes (processors)
    nodes = sorted(schedule.keys())
    
    # Group conditional tasks by their group ID for overlap detection
    conditional_groups = {}
    for node_schedule in schedule.values():
        for task in node_schedule:
            if task.is_conditional:
                group_id = task.conditional_group
                if group_id not in conditional_groups:
                    conditional_groups[group_id] = []
                conditional_groups[group_id].append(task)
    
    # Draw tasks using standard barh approach (matching saga.utils.draw.draw_gantt)
    for node in nodes:
        for task in schedule[node]:
            duration = task.end - task.start
            
            # Determine if this task overlaps with others in its conditional group
            is_overlapping = False
            if task.is_conditional:
                group_tasks = conditional_groups[task.conditional_group]
                # Check if any other task in the group overlaps in time
                for other_task in group_tasks:
                    if other_task.name != task.name:
                        # Check for time overlap
                        if not (task.end <= other_task.start or task.start >= other_task.end):
                            is_overlapping = True
                            break
            
            # Standard style: white fill with black border (matches codebase)
            # Overlapping conditional tasks get hatching to distinguish them
            if task.is_conditional and is_overlapping:
                ax.barh(
                    node, duration, left=task.start,
                    color='white', edgecolor='black',
                    hatch='//',  # Hatching for overlapping conditional tasks
                    linewidth=1.5
                )
            else:
                ax.barh(
                    node, duration, left=task.start,
                    color='white', edgecolor='black',
                    linewidth=1.5
                )
            
            # Add task label with group info for conditional tasks
            label_str = task.name
            if task.is_conditional:
                label_str += f"\n(G{task.conditional_group})"
            
            ax.text(
                task.start + duration / 2,
                node,
                label_str,
                ha='center',
                va='center',
                fontsize=20,  # Match standard font size
                color='black'
            )
    
    # Set up axes (standard style)
    ax.set_yticks(nodes)
    ax.set_yticklabels(nodes)
    ax.set_xlabel('Time')
    ax.set_ylabel('Node')
    ax.set_title(title, fontweight='bold')
    
    # Calculate makespan
    makespan = max(task.end for tasks in schedule.values() for task in tasks)
    ax.set_xlim(0, makespan * 1.1)  # Add 10% padding for makespan annotation
    
    # Add grid (standard style)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Add legend explaining the hatching (matching standard style)
    legend_elements = [
        mpatches.Patch(facecolor='white', edgecolor='black', label='Non-conditional task'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', 
                      label='Conditional task (overlapping)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add makespan annotation
    ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(makespan, nodes[-1], f'Makespan: {makespan:.2f}', 
            ha='right', va='top', fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nGantt chart saved to: {save_path}")
    
    return ax


def draw_schedule_comparison(original_schedule: Dict[Hashable, List[OverlappingTask]],
                             branch_schedules: Dict[str, Dict[Hashable, List[OverlappingTask]]],
                             task_graph: nx.DiGraph,
                             save_path: str = None):
    """Draw multiple schedules side-by-side for comparison.
    
    Shows the original schedule with overlapping tasks, plus individual schedules
    for each conditional branch.
    
    Args:
        original_schedule: The schedule with overlapping conditional tasks
        branch_schedules: Dict mapping branch names to their individual schedules
        task_graph: The task graph
        save_path: Optional path to save the figure
    """
    num_schedules = 1 + len(branch_schedules)
    fig, axes = plt.subplots(1, num_schedules, figsize=(6 * num_schedules, 6))
    
    if num_schedules == 1:
        axes = [axes]
    
    # Draw original schedule
    ax = axes[0]
    _draw_gantt_on_axis(ax, original_schedule, task_graph, 
                        "Overlapping Schedule\n(All branches)")
    
    # Draw individual branch schedules
    for idx, (branch_name, branch_schedule) in enumerate(branch_schedules.items(), 1):
        ax = axes[idx]
        _draw_gantt_on_axis(ax, branch_schedule, task_graph,
                           f"Branch: {branch_name}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison chart saved to: {save_path}")
    
    return axes


def _draw_gantt_on_axis(ax, schedule, task_graph, title):
    """Helper function to draw a Gantt chart on a specific axis.
    
    Uses the same style as saga.utils.draw.draw_gantt() for consistency.
    Conditional tasks are distinguished with hatching pattern.
    """
    nodes = sorted(schedule.keys())
    
    # Draw tasks using standard barh approach (matching saga.utils.draw.draw_gantt)
    for node in nodes:
        for task in schedule[node]:
            duration = task.end - task.start
            
            # Standard style: white fill with black border (matches codebase)
            # Conditional tasks get hatching to distinguish them
            if task.is_conditional:
                ax.barh(
                    node, duration, left=task.start,
                    color='white', edgecolor='black',
                    hatch='//',  # Hatching for conditional tasks
                    linewidth=1.5
                )
            else:
                ax.barh(
                    node, duration, left=task.start,
                    color='white', edgecolor='black',
                    linewidth=1.5
                )
            
            # Add task label in the center (standard approach)
            ax.text(
                task.start + duration / 2,
                node,
                task.name,
                ha='center',
                va='center',
                fontsize=20,  # Match standard font size
                color='black'
            )
    
    # Set up axes (standard style)
    ax.set_yticks(nodes)
    ax.set_yticklabels(nodes)
    ax.set_xlabel('Time')
    ax.set_ylabel('Node')
    ax.set_title(title, fontweight='bold')
    
    # Calculate makespan and set limits
    if schedule:
        makespan = max(task.end for tasks in schedule.values() for task in tasks if tasks)
        ax.set_xlim(0, makespan)
    
    # Add grid (standard style)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

