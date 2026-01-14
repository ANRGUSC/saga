"""Library of Conditional Task Graphs (CTGs) for testing overlapping scheduler.

Each CTG is defined as a function that returns:
    (ctg: nx.DiGraph, metadata: dict)
    
Metadata includes:
    - name: Short identifier
    - description: Brief description
    - ctg_type: Category (basic, nested, multi-branch, etc.)
    - num_branches: Number of possible execution paths
"""
import networkx as nx
from typing import Dict, List, Tuple, Any


def create_ctg0():
    """Basic 2-branch: A -> B/C -> D
    
    Structure:
        A
       / \
      B   C
       \ /
        D
    
    - Dashed edges (A→B, A→C) are conditional (mutually exclusive)
    - Solid edges (B→D, C→D) are unconditional
    """
    ctg = nx.DiGraph()
    
    ctg.add_node("A", weight=2)
    ctg.add_node("B", weight=3)
    ctg.add_node("C", weight=4)
    ctg.add_node("D", weight=2)
    
    ctg.add_edge("A", "B", weight=1, probability=0.75, conditional=True)
    ctg.add_edge("A", "C", weight=1, probability=0.25, conditional=True)
    ctg.add_edge("B", "D", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("C", "D", weight=1, probability=1.0, conditional=False)
    
    metadata = {
        "name": "ctg0",
        "description": "Basic 2-branch: A -> B/C -> D",
        "ctg_type": "basic",
        "num_branches": 2,
        "num_tasks": 4,
        "num_conditional_points": 1,
    }
    return ctg, metadata


def create_ctg1():
    """Mixed conditional/non-conditional: A -> B/C/D -> E
    
    Structure:
          A
        / | \
       B  C  D
        \ | /
          E
    
    - B and C are conditional alternatives (dashed, mutually exclusive)
    - D is always executed (solid edge from A)
    """
    ctg = nx.DiGraph()
    
    ctg.add_node("A", weight=2)
    ctg.add_node("B", weight=1)
    ctg.add_node("C", weight=3)
    ctg.add_node("D", weight=5)
    ctg.add_node("E", weight=1)
    
    ctg.add_edge("A", "B", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("A", "C", weight=1, probability=0.3, conditional=True)
    ctg.add_edge("A", "D", weight=1, probability=0.2, conditional=False)
    ctg.add_edge("B", "E", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("C", "E", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("D", "E", weight=1, probability=1.0, conditional=False)
    
    metadata = {
        "name": "ctg1",
        "description": "Mixed conditional/non-conditional: A -> B/C/D -> E",
        "ctg_type": "mixed",
        "num_branches": 2,  # B and C are conditional alternatives, D is always executed
        "num_tasks": 5,
        "num_conditional_points": 1,
    }
    return ctg, metadata


def create_ctg2():
    """Nested branches: A -> B/C, B -> D/E -> G
    
    Structure:
          A
         / \
        B   C
       / \   \
      D   E   |
      |   |   |
      F   |   |
       \  |  /
         \|/
          G
    
    - A→B, A→C: conditional (outer branch)
    - B→D, B→E: conditional (nested branch within B path)
    - All other edges: unconditional
    """
    ctg = nx.DiGraph()
    
    ctg.add_node("A", weight=2)
    ctg.add_node("B", weight=3)
    ctg.add_node("C", weight=4)
    ctg.add_node("D", weight=2)
    ctg.add_node("E", weight=3)
    ctg.add_node("F", weight=2)
    ctg.add_node("G", weight=2)
    
    ctg.add_edge("A", "B", weight=1, probability=0.75, conditional=True)
    ctg.add_edge("A", "C", weight=1, probability=0.25, conditional=True)
    ctg.add_edge("B", "D", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("B", "E", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("D", "F", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("F", "G", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("E", "G", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("C", "G", weight=1, probability=1.0, conditional=False)
    
    metadata = {
        "name": "ctg2",
        "description": "Nested branches: A -> B/C, B -> D/E -> G",
        "ctg_type": "nested",
        "num_branches": 3,  # B-D-F-G, B-E-G, C-G
        "num_tasks": 7,
        "num_conditional_points": 2,
    }
    return ctg, metadata


def create_ctg3():
    """Symmetric nested: A -> B/C, B -> D/E, C -> F/G -> I
    
    Structure:
            A
           / \
          B   C
         / \ / \
        D  E F  G
        |   \|/
        H   /
         \ /
          I
    
    - A→B, A→C: conditional (outer branch)
    - B→D, B→E: conditional (nested within B)
    - C→F, C→G: conditional (nested within C)
    - All paths converge at I
    """
    ctg = nx.DiGraph()
    
    ctg.add_node("A", weight=2)
    ctg.add_node("B", weight=3)
    ctg.add_node("C", weight=4)
    ctg.add_node("D", weight=2)
    ctg.add_node("E", weight=3)
    ctg.add_node("F", weight=2)
    ctg.add_node("G", weight=3)
    ctg.add_node("H", weight=2)
    ctg.add_node("I", weight=2)
    
    ctg.add_edge("A", "B", weight=1, probability=0.75, conditional=True)
    ctg.add_edge("A", "C", weight=1, probability=0.25, conditional=True)
    ctg.add_edge("B", "D", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("B", "E", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("C", "F", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("C", "G", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("D", "H", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("H", "I", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("E", "I", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("F", "I", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("G", "I", weight=1, probability=1.0, conditional=False)
    
    metadata = {
        "name": "ctg3",
        "description": "Symmetric nested: A -> B/C, B -> D/E, C -> F/G -> I",
        "ctg_type": "symmetric_nested",
        "num_branches": 4,  # B-D-H-I, B-E-I, C-F-I, C-G-I
        "num_tasks": 9,
        "num_conditional_points": 3,
    }
    return ctg, metadata


def create_ctg4():
    """Post-convergence branching: nested + branch after convergence
    
    Structure:
            A
           / \
          B   C
         / \ / \
        D  E F  G
        |   \|/
        H   /
         \ /
          I
         / \
        J   K
    
    - Same as ctg3, but with additional branch after convergence at I
    - I→J, I→K: conditional (post-convergence branch)
    """
    ctg = nx.DiGraph()
    
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
    
    ctg.add_edge("A", "B", weight=1, probability=0.75, conditional=True)
    ctg.add_edge("A", "C", weight=1, probability=0.25, conditional=True)
    ctg.add_edge("B", "D", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("B", "E", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("C", "F", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("C", "G", weight=1, probability=0.5, conditional=True)
    ctg.add_edge("D", "H", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("H", "I", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("E", "I", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("F", "I", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("G", "I", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("I", "J", weight=1, probability=0.6, conditional=True)
    ctg.add_edge("I", "K", weight=1, probability=0.4, conditional=True)
    
    metadata = {
        "name": "ctg4",
        "description": "Post-convergence branching: nested + branch after convergence",
        "ctg_type": "post_convergence",
        "num_branches": 8,  # 4 paths to I * 2 paths from I
        "num_tasks": 11,
        "num_conditional_points": 4,
    }
    return ctg, metadata


def create_ctg5():
    """Multi-branch (4-way): A -> B/C/D/E -> F
    
    Structure:
           A
        /|\ \
       B C D E
        \|/ /
          F
    
    - All edges from A are conditional (4-way mutually exclusive)
    - All edges to F are unconditional
    """
    ctg = nx.DiGraph()
    
    ctg.add_node("A", weight=2)
    ctg.add_node("B", weight=3)
    ctg.add_node("C", weight=4)
    ctg.add_node("D", weight=2)
    ctg.add_node("E", weight=5)
    ctg.add_node("F", weight=2)
    
    ctg.add_edge("A", "B", weight=1, probability=0.4, conditional=True)
    ctg.add_edge("A", "C", weight=1, probability=0.3, conditional=True)
    ctg.add_edge("A", "D", weight=1, probability=0.2, conditional=True)
    ctg.add_edge("A", "E", weight=1, probability=0.1, conditional=True)
    ctg.add_edge("B", "F", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("C", "F", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("D", "F", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("E", "F", weight=1, probability=1.0, conditional=False)
    
    metadata = {
        "name": "ctg5",
        "description": "Multi-branch (4-way): A -> B/C/D/E -> F",
        "ctg_type": "multi_branch",
        "num_branches": 4,
        "num_tasks": 6,
        "num_conditional_points": 1,
    }
    return ctg, metadata


def create_ctg6():
    """Deep chain with branch: long sequential + branch in middle
    
    Structure:
        A → B → C
               / \
              D   E
               \ /
                F → G → H
    
    - A→B, B→C: unconditional (sequential prefix)
    - C→D, C→E: conditional (branch in middle)
    - D→F, E→F, F→G, G→H: unconditional (sequential suffix)
    """
    ctg = nx.DiGraph()
    
    # Long chain: A -> B -> C -> branch -> F -> G -> H
    ctg.add_node("A", weight=2)
    ctg.add_node("B", weight=2)
    ctg.add_node("C", weight=2)
    ctg.add_node("D", weight=3)  # Branch option 1
    ctg.add_node("E", weight=4)  # Branch option 2
    ctg.add_node("F", weight=2)
    ctg.add_node("G", weight=2)
    ctg.add_node("H", weight=2)
    
    ctg.add_edge("A", "B", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("B", "C", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("C", "D", weight=1, probability=0.6, conditional=True)
    ctg.add_edge("C", "E", weight=1, probability=0.4, conditional=True)
    ctg.add_edge("D", "F", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("E", "F", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("F", "G", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("G", "H", weight=1, probability=1.0, conditional=False)
    
    metadata = {
        "name": "ctg6",
        "description": "Deep chain with branch: A-B-C -> D/E -> F-G-H",
        "ctg_type": "deep_chain",
        "num_branches": 2,
        "num_tasks": 8,
        "num_conditional_points": 1,
    }
    return ctg, metadata


def create_ctg7():
    """Imbalanced branches: one path much longer than another
    
    Structure:
        A
       / \
      B   C
      |   |
      |   D
      |   |
      |   E
      |   |
      |   F
       \ /
        G
    
    - Short path: A → B → G (2 hops)
    - Long path: A → C → D → E → F → G (5 hops)
    - A→B, A→C: conditional
    """
    ctg = nx.DiGraph()
    
    ctg.add_node("A", weight=2)
    ctg.add_node("B", weight=2)  # Short path
    ctg.add_node("C", weight=2)  # Long path start
    ctg.add_node("D", weight=2)
    ctg.add_node("E", weight=2)
    ctg.add_node("F", weight=2)  # Long path end
    ctg.add_node("G", weight=2)  # Convergence
    
    # Short path: A -> B -> G
    ctg.add_edge("A", "B", weight=1, probability=0.3, conditional=True)
    ctg.add_edge("B", "G", weight=1, probability=1.0, conditional=False)
    
    # Long path: A -> C -> D -> E -> F -> G
    ctg.add_edge("A", "C", weight=1, probability=0.7, conditional=True)
    ctg.add_edge("C", "D", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("D", "E", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("E", "F", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("F", "G", weight=1, probability=1.0, conditional=False)
    
    metadata = {
        "name": "ctg7",
        "description": "Imbalanced branches: short (A-B-G) vs long (A-C-D-E-F-G)",
        "ctg_type": "imbalanced",
        "num_branches": 2,
        "num_tasks": 7,
        "num_conditional_points": 1,
    }
    return ctg, metadata


def create_ctg8():
    """Wide fan-out: multiple tasks after single branch point
    
    Structure:
            A
           / \
          B   C
         / \ / \
        D  E F  G
         \ | | /
          \| |/
            H
    
    - A→B, A→C: conditional (mutually exclusive branches)
    - B fans out to D and E in parallel (unconditional)
    - C fans out to F and G in parallel (unconditional)
    - All paths converge at H
    """
    ctg = nx.DiGraph()
    
    ctg.add_node("A", weight=2)
    ctg.add_node("B", weight=3)  # Branch 1
    ctg.add_node("C", weight=4)  # Branch 2
    ctg.add_node("D", weight=2)  # After B
    ctg.add_node("E", weight=2)  # After B
    ctg.add_node("F", weight=2)  # After C
    ctg.add_node("G", weight=2)  # After C
    ctg.add_node("H", weight=2)  # Convergence
    
    ctg.add_edge("A", "B", weight=1, probability=0.6, conditional=True)
    ctg.add_edge("A", "C", weight=1, probability=0.4, conditional=True)
    # B fans out to D and E (parallel)
    ctg.add_edge("B", "D", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("B", "E", weight=1, probability=1.0, conditional=False)
    # C fans out to F and G (parallel)
    ctg.add_edge("C", "F", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("C", "G", weight=1, probability=1.0, conditional=False)
    # All converge to H
    ctg.add_edge("D", "H", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("E", "H", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("F", "H", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("G", "H", weight=1, probability=1.0, conditional=False)
    
    metadata = {
        "name": "ctg8",
        "description": "Wide fan-out: A -> B/C, B -> D+E, C -> F+G -> H",
        "ctg_type": "wide_fanout",
        "num_branches": 2,
        "num_tasks": 8,
        "num_conditional_points": 1,
    }
    return ctg, metadata


def create_ctg9():
    """Heavy conditional task: one branch has much heavier computation
    
    Structure:
        A
       / \
      B   C
       \ /
        D
    
    - Same structure as ctg0
    - B has weight=2 (light path, 80% probability)
    - C has weight=20 (heavy path, 20% probability)
    """
    ctg = nx.DiGraph()
    
    ctg.add_node("A", weight=2)
    ctg.add_node("B", weight=2)   # Light path
    ctg.add_node("C", weight=20)  # Heavy path (10x heavier!)
    ctg.add_node("D", weight=2)
    
    ctg.add_edge("A", "B", weight=1, probability=0.8, conditional=True)
    ctg.add_edge("A", "C", weight=1, probability=0.2, conditional=True)
    ctg.add_edge("B", "D", weight=1, probability=1.0, conditional=False)
    ctg.add_edge("C", "D", weight=1, probability=1.0, conditional=False)
    
    metadata = {
        "name": "ctg9",
        "description": "Heavy conditional: B(w=2) vs C(w=20)",
        "ctg_type": "heavy_conditional",
        "num_branches": 2,
        "num_tasks": 4,
        "num_conditional_points": 1,
    }
    return ctg, metadata

# Registry of all CTGs

def get_all_ctgs():
    """Return list of all CTGs with their metadata."""
    return [
        create_ctg0(),
        create_ctg1(),
        create_ctg2(),
        create_ctg3(),
        create_ctg4(),
        create_ctg5(),
        create_ctg6(),
        create_ctg7(),
        create_ctg8(),
        create_ctg9(),
    ]


def get_ctg_by_name(name: str):
    """Get a specific CTG by name."""
    for ctg, metadata in get_all_ctgs():
        if metadata["name"] == name:
            return ctg, metadata


def get_ctgs_by_type(ctg_type: str):
    """Get all CTGs of a specific type."""
    result = []
    for ctg, meta in get_all_ctgs():
        if meta["ctg_type"] == ctg_type:
            result.append((ctg, meta))
    return result



#Network definitions

def create_simple_network(num_nodes: int = 2) -> nx.Graph:
    """Create a simple N-node network with uniform properties.
    
    Args:
        num_nodes: Number of compute nodes (default 2)
        
    Returns:
        NetworkX Graph with nodes and edges configured for scheduling
    """
    network = nx.Graph()
    nodes = list(range(1, num_nodes + 1))
    network.add_nodes_from(nodes, weight=1.0)  # All nodes have speed 1.0
    
    # Add edges between all pairs (fully connected)
    for i in nodes:
        for j in nodes:
            if i < j:
                network.add_edge(i, j, weight=1.0)  # Link bandwidth 1.0
    
    # Add self-loops with very high bandwidth (no cost for same-node communication)
    for node in nodes:
        network.add_edge(node, node, weight=1e9)
    
    return network


def create_heterogeneous_network() -> nx.Graph:
    """Create a 3-node heterogeneous network with different speeds."""
    network = nx.Graph()
    
    # Nodes with different speeds
    network.add_node(1, weight=1.0)   # Normal speed
    network.add_node(2, weight=2.0)   # 2x faster
    network.add_node(3, weight=0.5)   # 2x slower
    
    # Different link bandwidths
    network.add_edge(1, 2, weight=2.0)   # Fast link
    network.add_edge(1, 3, weight=0.5)   # Slow link
    network.add_edge(2, 3, weight=1.0)   # Normal link
    
    # Self-loops
    for node in [1, 2, 3]:
        network.add_edge(node, node, weight=1e9)
    
    return network

