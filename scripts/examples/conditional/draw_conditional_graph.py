"""Custom drawing function for conditional task graphs."""
import matplotlib.pyplot as plt
import networkx as nx
from saga.utils.draw import draw_task_graph


def draw_conditional_task_graph(dag: nx.DiGraph, 
                                filename: str, 
                                use_latex: bool = False,
                                **kwargs):
    """
    Draw a conditional task graph with visual distinction for conditional edges.
    
    Conditional edges (where conditional=True) are shown:
    - With dashed lines
    - With probability in the edge label
    
    Args:
        dag: The conditional task graph
        filename: Output filename (will add .pdf and .png extensions)
        use_latex: Whether to use LaTeX for rendering. Defaults to False.
        **kwargs: Additional arguments to pass to draw_task_graph
    """
    # Set default kwargs
    default_kwargs = {
        'figsize': (12, 8),
        'node_size': 2000,
        'font_size': 16,
        'draw_node_weights': True,
        'draw_edge_weights': False,  # We'll draw custom edge labels
    }
    default_kwargs.update(kwargs)
    
    # Create a copy of the graph to modify for display
    display_dag = dag.copy()
    
    # Add probability to edge labels for conditional edges
    for u, v in display_dag.edges:
        edge_data = display_dag.edges[u, v]
        weight = edge_data.get('weight', 0)
        prob = edge_data.get('probability', 1.0)
        is_conditional = edge_data.get('conditional', False)
        
        if is_conditional:
            # Include probability in label
            if use_latex:
                label = r"$c=%s, p=%s$" % (round(weight, 1), round(prob, 2))
            else:
                label = f"c={round(weight, 1)}, p={round(prob, 2)}"
        else:
            # Regular edge label
            if use_latex:
                label = r"$c=%s$" % round(weight, 1)
            else:
                label = f"c={round(weight, 1)}"
        
        display_dag.edges[u, v]['label'] = label
    
    # Draw the base graph
    ax = draw_task_graph(display_dag, use_latex=use_latex, **default_kwargs)
    
    # Now we need to redraw edges with different styles for conditional ones
    # Get the position layout that was used
    pos = nx.nx_agraph.graphviz_layout(display_dag, prog="dot")
    
    # Separate conditional and non-conditional edges
    conditional_edges = [(u, v) for u, v in display_dag.edges 
                         if display_dag.edges[u, v].get('conditional', False)]
    non_conditional_edges = [(u, v) for u, v in display_dag.edges 
                             if not display_dag.edges[u, v].get('conditional', False)]
    
    # Redraw edges with different styles
    # First draw non-conditional edges (solid)
    if non_conditional_edges:
        nx.draw_networkx_edges(
            display_dag, pos=pos, ax=ax,
            edgelist=non_conditional_edges,
            arrowsize=20, arrowstyle="->",
            width=2, edge_color="black",
            style='solid',
            node_size=default_kwargs['node_size'],
        )
    
    # Then draw conditional edges (dashed, red)
    if conditional_edges:
        nx.draw_networkx_edges(
            display_dag, pos=pos, ax=ax,
            edgelist=conditional_edges,
            arrowsize=20, arrowstyle="->",
            width=2, edge_color="red",
            style='dashed',
            node_size=default_kwargs['node_size'],
        )
    
    # Draw edge labels
    edge_labels = {}
    for u, v in display_dag.edges:
        if 'label' in display_dag.edges[u, v]:
            edge_labels[(u, v)] = display_dag.edges[u, v]['label']
    
    nx.draw_networkx_edge_labels(
        display_dag, pos=pos, ax=ax,
        edge_labels=edge_labels,
        font_size=default_kwargs.get('weight_font_size', 12),
    )
    
    # Save as both PDF and PNG
    ax.get_figure().savefig(f'{filename}.pdf', bbox_inches='tight')
    ax.get_figure().savefig(f'{filename}.png', bbox_inches='tight', dpi=300)
    
    # Close to free memory
    plt.close(ax.get_figure())
    
    print("  Red dashed edges, conditional branches")

