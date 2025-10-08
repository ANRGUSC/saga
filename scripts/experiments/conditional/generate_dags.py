"""
Generate conditional tree DAGs for testing.
"""
import networkx as nx
import random
from typing import List, Dict, Tuple

def create_conditional_tree_dag(levels: int = 3, branching_factor: int = 2, conditional_probabilities: List[float] = [0.6, 0.4]) -> nx.DiGraph:
    """
    Create a tree DAG with conditional branches at each level.

    Structure:
    - Level 0: Root node
    - Level 1: test with deterministic 
    - Level 2+: Conditional branches with probabilities

    TODO: implement the branching factor 
    """
    dag = nx.DiGraph()
    
    # Add root node
    dag.add_node("A", weight=1, level=0, conditional=False)
    
    # Level 1: Deterministic branching
    level1_nodes = ["B", "C"]
    for node in level1_nodes:
        dag.add_node(node, weight=2, level=1, conditional=False)
        dag.add_edge("A", node, weight=1, probability=1.0, conditional=False)
    
 # Level 2+: Conditional branching
    current_level_nodes = level1_nodes
    node_counter = 0
    
    for level in range(2, levels + 1):
        next_level_nodes = []
        
        for parent in current_level_nodes:
            # Create conditional children
            #enumerate turns [0.6, 0.4] -> into i=0 prob = 0.6, i=1 prob = 0.4.
            for i, prob in enumerate(conditional_probabilities):
                child = f"L{level}_{node_counter}"
                node_counter += 1
                
                dag.add_node(child, 
                           weight=random.uniform(0.2, 2), 
                           level=level, 
                           conditional=True,
                           probability=prob)
                dag.add_edge(parent, child, 
                           weight=random.uniform(0.2, 1.0), 
                           probability=prob, 
                           conditional=True)
                next_level_nodes.append(child)
        
        current_level_nodes = next_level_nodes
    
    return dag


def create_test_conditional_dags() -> List[Tuple[str, nx.DiGraph]]:
    """
    Create various test cases for conditional tree DAGs
    
    Return: 
        casename: Name of the dag
        dag: the actual dag 
    """
    test_cases = []
    
    # Case 1: Simple 2-level conditional tree
    dag1 = create_conditional_tree_dag(levels=2, branching_factor=2, 
                                     conditional_probabilities=[0.7, 0.3])
    test_cases.append(("2level", dag1))
    
    # Case 2: 3-level tree with different probabilities
    dag2 = create_conditional_tree_dag(levels=3, branching_factor=2, 
                                     conditional_probabilities=[0.8, 0.2])
    test_cases.append(("3level", dag2))
    
    # Case 3: Binary tree with equal probabilities
    dag3 = create_conditional_tree_dag(levels=4, branching_factor=2, 
                                     conditional_probabilities=[0.5, 0.5])
    test_cases.append(("4level", dag3))
    
    return test_cases


def save_dag_as_pdf(dag: nx.DiGraph, filename: str, use_latex: bool = False, **kwargs):
    """
    Simple helper to save a networkx DiGraph as a PDF using saga's draw_task_graph
    
    Args:
        dag: The networkx DiGraph to visualize
        filename: Output filename (will add .pdf and .png extensions)
        use_latex: Whether to use LaTeX for rendering. Defaults to False.
        **kwargs: Additional arguments to pass to draw_task_graph (e.g., figsize, node_size, font_size)
    
    Example:
        >>> dag = create_conditional_tree_dag(levels=2, branching_factor=2)
        >>> save_dag_as_pdf(dag, "my_dag", figsize=(10, 6), node_size=2000)
    """
    from saga.utils.draw import draw_task_graph
    import matplotlib.pyplot as plt
    
    # Set default kwargs
    default_kwargs = {
        'figsize': (12, 8),
        'node_size': 2000,
        'font_size': 16,
        'draw_node_weights': True,
        'draw_edge_weights': True
    }
    default_kwargs.update(kwargs)
    
    # Draw the graph
    ax = draw_task_graph(dag, use_latex=use_latex, **default_kwargs)
    
    # Save as both PDF and PNG

    ax.get_figure().savefig(f'{filename}.png', bbox_inches='tight', dpi=300)
    
    # Close to free memory
    plt.close(ax.get_figure())
    
    print(f"Saved: {filename}.pdf and {filename}.png")