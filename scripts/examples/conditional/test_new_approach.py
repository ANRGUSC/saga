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
    extract_all_branches_with_recalculation,
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


def create_simple_ctg2():
    """Create CTG with nested conditional branches: A -> B/C, B -> D/E -> F/G -> G
    
    Structure:
           A
          / \
         B   C        <- Branch 1: A chooses B or C
        / \   \
       D   E   |      <- Branch 2: B chooses D or E (only if B was chosen)
       |   |   |
       F   |   |      <- D leads to F
        \  |  /
         \ | /
           G          <- All paths converge to G
    
    Possible execution paths:
        Path 1: A -> B -> D -> F -> G
        Path 2: A -> B -> E -> G
        Path 3: A -> C -> G
    
    Node weights:
        A: weight=2
        B: weight=3, conditional from A, prob=0.75
        C: weight=4, conditional from A, prob=0.25
        D: weight=2, conditional from B, prob=0.5
        E: weight=3, conditional from B, prob=0.5
        F: weight=2
        G: weight=2
    
    Communication times = 1
    """
    ctg = nx.DiGraph()
    
    # Add nodes
    ctg.add_node("A", weight=2)
    ctg.add_node("B", weight=3)
    ctg.add_node("C", weight=4)
    ctg.add_node("D", weight=2)
    ctg.add_node("E", weight=3)
    ctg.add_node("F", weight=2)
    ctg.add_node("G", weight=2)
    
    # Add edges
    # Branch 1: A branches to B or C (conditional)
    ctg.add_edge("A", "B", weight=1, probability=0.75, conditional=True)
    ctg.add_edge("A", "C", weight=1, probability=0.25, conditional=True)
    
    # Branch 2: B branches to D or E (conditional, nested)
    ctg.add_edge("B", "D", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("B", "E", weight=1, probability=0.5, conditional=True)
    
    # D leads to F (non-conditional)
    ctg.add_edge("D", "F", weight=1, probability=1.0, conditional=False)
    
    # All paths converge to G (non-conditional)
    ctg.add_edge("F", "G", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("E", "G", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("C", "G", weight=1, probability=1.0, conditional=False)
    
    return ctg


def create_simple_ctg3():
    """Create CTG with symmetric nested conditional branches on both sides.
    
    Structure:
           A
          / \
         B   C        <- Branch 1: A chooses B or C
        / \ / \
       D  E F  G      <- Branch 2: B chooses D or E, C chooses F or G
       |  | |  |
       H  | |  |      <- D leads to H
        \ | | /
         \| |/
           I          <- All paths converge to I
    
    Possible execution paths:
        Path 1: A -> B -> D -> H -> I
        Path 2: A -> B -> E -> I
        Path 3: A -> C -> F -> I
        Path 4: A -> C -> G -> I
    
    Node weights:
        A: weight=2
        B: weight=3, conditional from A, prob=0.75
        C: weight=4, conditional from A, prob=0.25
        D: weight=2, conditional from B, prob=0.5
        E: weight=3, conditional from B, prob=0.5
        F: weight=2, conditional from C, prob=0.5
        G: weight=3, conditional from C, prob=0.5
        H: weight=2
        I: weight=2
    
    Communication times = 1
    """
    ctg = nx.DiGraph()
    
    # Add nodes
    ctg.add_node("A", weight=2)
    ctg.add_node("B", weight=3)
    ctg.add_node("C", weight=4)
    ctg.add_node("D", weight=2)
    ctg.add_node("E", weight=3)
    ctg.add_node("F", weight=2)
    ctg.add_node("G", weight=3)
    ctg.add_node("H", weight=2)
    ctg.add_node("I", weight=2)
    
    # Add edges
    # Branch 1: A branches to B or C (conditional)
    ctg.add_edge("A", "B", weight=1, probability=0.75, conditional=True)
    ctg.add_edge("A", "C", weight=1, probability=0.25, conditional=True)
    
    # Branch 2: B branches to D or E (conditional, nested)
    ctg.add_edge("B", "D", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("B", "E", weight=1, probability=0.5, conditional=True)
    
    # Branch 3: C branches to F or G (conditional, nested)
    ctg.add_edge("C", "F", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("C", "G", weight=1, probability=0.5, conditional=True)
    
    # D leads to H (non-conditional)
    ctg.add_edge("D", "H", weight=1, probability=1.0, conditional=False)
    
    # All paths converge to I (non-conditional)
    ctg.add_edge("H", "I", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("E", "I", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("F", "I", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("G", "I", weight=1, probability=1.0, conditional=False)
    
    return ctg


def create_simple_ctg4():
    """Create CTG with branching at multiple levels including after convergence.
    
    Structure:
             A
            / \
           B   C        <- Branch 1: A chooses B or C
          / \ / \
         D  E F  G      <- Branch 2: B chooses D or E, C chooses F or G
         |  | |  |
         H  | |  |      <- D leads to H
          \ | | /
           \| |/
             I          <- All paths converge to I
            / \
           J   K        <- Branch 3: I chooses J or K (after convergence!)
    
    Possible execution paths (8 total):
        Path 1: A -> B -> D -> H -> I -> J
        Path 2: A -> B -> D -> H -> I -> K
        Path 3: A -> B -> E -> I -> J
        Path 4: A -> B -> E -> I -> K
        Path 5: A -> C -> F -> I -> J
        Path 6: A -> C -> F -> I -> K
        Path 7: A -> C -> G -> I -> J
        Path 8: A -> C -> G -> I -> K
    
    Node weights:
        A: weight=2
        B: weight=3, conditional from A, prob=0.75
        C: weight=4, conditional from A, prob=0.25
        D: weight=2, conditional from B, prob=0.5
        E: weight=3, conditional from B, prob=0.5
        F: weight=2, conditional from C, prob=0.5
        G: weight=3, conditional from C, prob=0.5
        H: weight=2
        I: weight=2
        J: weight=3, conditional from I, prob=0.6
        K: weight=2, conditional from I, prob=0.4
    
    Communication times = 1
    """
    ctg = nx.DiGraph()
    
    # Add nodes
    ctg.add_node("A", weight=2)
    ctg.add_node("B", weight=3)
    ctg.add_node("C", weight=4)
    ctg.add_node("D", weight=2)
    ctg.add_node("E", weight=3)
    ctg.add_node("F", weight=2)
    ctg.add_node("G", weight=3)
    ctg.add_node("H", weight=2)
    ctg.add_node("I", weight=2)
    ctg.add_node("J", weight=3)
    ctg.add_node("K", weight=2)
    
    # Add edges
    # Branch 1: A branches to B or C (conditional)
    ctg.add_edge("A", "B", weight=1, probability=0.75, conditional=True)
    ctg.add_edge("A", "C", weight=1, probability=0.25, conditional=True)
    
    # Branch 2: B branches to D or E (conditional, nested)
    ctg.add_edge("B", "D", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("B", "E", weight=1, probability=0.5, conditional=True)
    
    # Branch 3: C branches to F or G (conditional, nested)
    ctg.add_edge("C", "F", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("C", "G", weight=1, probability=0.5, conditional=True)
    
    # D leads to H (non-conditional)
    ctg.add_edge("D", "H", weight=1, probability=1.0, conditional=False)
    
    # All paths converge to I (non-conditional)
    ctg.add_edge("H", "I", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("E", "I", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("F", "I", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("G", "I", weight=1, probability=1.0, conditional=False)
    
    # Branch 4: I branches to J or K (conditional, after convergence!)
    ctg.add_edge("I", "J", weight=1, probability=0.6, conditional=True)
    ctg.add_edge("I", "K", weight=1, probability=0.4, conditional=True)
    
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
    ctg = create_simple_ctg3()
    
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
    
    # Get runtimes and commtimes (scheduler-independent - can be used by branch extractor)
    runtimes, commtimes = scheduler.get_runtimes(network, ctg)
    
    # Create schedule
    schedule = scheduler.schedule(network, ctg)
    
    # Visualize the schedule
    ax = draw_overlapping_gantt(schedule, ctg, 
                                 title="Simple CTG: A -> B/C -> D with Overlapping",
                                 save_path="overlapping_schedule.pdf")
    
    # Extract branches with recalculated times (removes gaps)
    # Pass runtimes and commtimes instead of network - makes branch_extractor scheduler-independent
    branch_schedules_recalculated = extract_all_branches_with_recalculation(
        schedule, ctg, runtimes, commtimes
    )

    
    # Visualize comparison 
    axes = draw_schedule_comparison(schedule, branch_schedules_recalculated, ctg,
                                    save_path="schedule_comparison.pdf")
    

if __name__ == "__main__":
    main()

