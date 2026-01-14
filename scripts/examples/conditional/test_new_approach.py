"""Test new conditional scheduling approach where conditional tasks overlap.

This script demonstrates the workflow for a single CTG. For batch testing
of all CTGs, use run_all_ctgs.py instead.
"""
import sys
import pathlib
import matplotlib.pyplot as plt

from overlapping_scheduler import (
    OverlappingConditionalScheduler,
    OverlappingTask
)
from visualize_overlapping import draw_overlapping_gantt, draw_schedule_comparison
from branch_extractor import (
    extract_all_branches_with_recalculation,
    generate_heft_comparison_schedules,
)
from saga.utils.draw import draw_network
from draw_conditional_graph import draw_conditional_task_graph

# Import CTGs from the library
from ctg_library import (
    get_all_ctgs,
    get_ctg_by_name,
    create_simple_network,
    create_heterogeneous_network,
    create_ctg0,
    create_ctg1,
    create_ctg2,
    create_ctg3,
    create_ctg4,
    create_ctg5,
)

def main():
    """Demonstrate the overlapping scheduler workflow on a single CTG."""
    # Select which CTG to test (change this to test different CTGs)
    ctg_name = "ctg4"  # Options: ctg0-ctg9
    
    # Get CTG from library
    ctg, metadata = get_ctg_by_name(ctg_name)
    
    # Create network
    network = create_simple_network(2)
    
    # Visualize the CTG structure with conditional edges highlighted
    draw_conditional_task_graph(ctg, "simple_ctg")
    
    # Visualize the network structure
    ax_network = draw_network(network, use_latex=False, draw_colors=True, figsize=(8, 6))
    ax_network.get_figure().savefig('simple_network.pdf', bbox_inches='tight')
    ax_network.get_figure().savefig('simple_network.png', bbox_inches='tight', dpi=300)
    plt.close(ax_network.get_figure())
    
    # Schedule using new overlapping approach
    scheduler = OverlappingConditionalScheduler()
    
    # Get runtimes and commtimes (scheduler-independent - can be used by branch extractor)
    runtimes, commtimes = scheduler.get_runtimes(network, ctg)
    
    # Create schedule
    schedule = scheduler.schedule(network, ctg)
    
    # Visualize the schedule
    ax = draw_overlapping_gantt(schedule, ctg, 
                                 title=f"{metadata['name']}: {metadata['description']}",
                                 save_path="overlapping_schedule.pdf")
    
    # Extract branches with recalculated times (removes gaps)
    # Pass runtimes and commtimes instead of network - makes branch_extractor scheduler-independent
    branch_schedules_recalculated = extract_all_branches_with_recalculation(
        schedule, ctg, runtimes, commtimes
    )
    
    # Visualize comparison 
    axes = draw_schedule_comparison(schedule, branch_schedules_recalculated, ctg,
                                    save_path="schedule_comparison.pdf")
    
    # Generate HEFT comparison schedules (standard HEFT on each branch independently)
    heft_schedules = generate_heft_comparison_schedules(ctg, network)
    
    # Visualize HEFT comparison
    axes_heft = draw_schedule_comparison(schedule, heft_schedules, ctg,
                                         save_path="heft_comparison.pdf")

    print("test/Done")


if __name__ == "__main__":
    main()

