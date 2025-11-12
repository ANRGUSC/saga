"""Test new conditional scheduling approach where conditional tasks overlap."""
import sys
import pathlib
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


from overlapping_scheduler import (
    OverlappingConditionalScheduler,
    OverlappingTask
)
from visualize_overlapping import draw_overlapping_gantt, draw_schedule_comparison
from branch_extractor import (
    extract_all_branches,
    print_branch_schedules,
    verify_branch_consistency
)
from saga.utils.draw import draw_network
from draw_conditional_graph import draw_conditional_task_graph

def create_simple_ctg():
    """Create simple test CTG: A -> B/C -> D
    
    A: weight=2
    B: weight=3, conditional, prob=3/4
    C: weight=4, conditional, prob=1/4
    D: weight=2
    Communication times = 1
    """
    ctg = nx.DiGraph()
    
    # Add nodes
    ctg.add_node("A", weight=2)
    ctg.add_node("B", weight=3)
    ctg.add_node("C", weight=4)
    ctg.add_node("D", weight=2)
    
    # Add edges
    # A branches to B or C (conditional)
    ctg.add_edge("A", "B", weight=1, probability=0.75, conditional=True)
    ctg.add_edge("A", "C", weight=1, probability=0.25, conditional=True)
    
    # B and C both lead to D (non-conditional from each branch)
    ctg.add_edge("B", "D", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("C", "D", weight=1, probability=1.0, conditional=False)
    
    return ctg

def create_network():
    """Create a simple 2-node network."""
    network = nx.Graph()
    network.add_nodes_from([1, 2], weight=1.0)  # Both nodes have speed 1.0
    network.add_edges_from([(1, 2)], weight=1.0)  # Link has bandwidth 1.0
    # Add self-loops with very high bandwidth (no cost for same-node communication)
    network.add_edges_from([(1, 1), (2, 2)], weight=1e9)
    return network

def main():
    # Create simple CTG
    ctg = create_simple_ctg()
    
    # Create network
    network = create_network()
    
    # Visualize the CTG structure with conditional edges highlighted
    draw_conditional_task_graph(ctg, "simple_ctg")
    
    # Visualize the network structure
    ax_network = draw_network(network, use_latex=False, draw_colors=True, figsize=(8, 6))
    ax_network.get_figure().savefig('simple_network.pdf', bbox_inches='tight')
    ax_network.get_figure().savefig('simple_network.png', bbox_inches='tight', dpi=300)
    plt.close(ax_network.get_figure())
    
    # Schedule using new overlapping approach
    
    scheduler = OverlappingConditionalScheduler()

    schedule = scheduler.schedule(network, ctg)

    """
    schedule = {
        "node1": [
            OverlappingTask(node="node1", name="A", start=0, end=2, group=-1, is_conditional=False),
            OverlappingTask(node="node1", name="B", start=2, end=5, group=0, is_conditional=True),
            OverlappingTask(node="node1", name="C", start=2, end=6, group=0, is_conditional=True),  # ← Overlaps with B!
            OverlappingTask(node="node1", name="D", start=7, end=9, group=-1, is_conditional=False)
        ],
        "node2": []
    }
    """
    # Visualize the schedule
  
    ax = draw_overlapping_gantt(schedule, ctg, 
                                 title="Simple CTG: A -> B/C -> D with Overlapping",
                                 save_path="overlapping_schedule.pdf")
    
    # Extract individual branch schedules

    
    branch_schedules = extract_all_branches(schedule, ctg)
    
    """
        branch_schedules = {
            "Path: B": {  # ← Branch name (key)
                "node1": [  # ← Node name (key)
                    OverlappingTask(node="node1", name="A", start=0, end=2, ...),  # ← List of tasks
                    OverlappingTask(node="node1", name="B", start=2, end=5, ...),
                    OverlappingTask(node="node1", name="D", start=7, end=9, ...)
                    # ← C removed!
                ],
                "node2": []
            },
            "Path: C": {  # ← Branch name (key)
                "node1": [  # ← Node name (key)
                    OverlappingTask(node="node1", name="A", start=0, end=2, ...),  # ← List of tasks
                    OverlappingTask(node="node1", name="C", start=2, end=6, ...),
                    OverlappingTask(node="node1", name="D", start=7, end=9, ...)
                    # ← B removed!
                ],
                "node2": []
            }
        }
    """
    
    #print_branch_schedules(branch_schedules)
    
    # Verify consistency
    #verify_branch_consistency(branch_schedules, ctg)
    
    # Visualize comparison

    
    axes = draw_schedule_comparison(schedule, branch_schedules, ctg,
                                    save_path="schedule_comparison.pdf")
    

if __name__ == "__main__":
    main()

