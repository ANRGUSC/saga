from typing import Dict, Hashable, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from saga.scheduler import Task

from ..utils.random_variable import RandomVariable


def format_graph(graph: nx.DiGraph) -> str:
    """Formats the graph

    copies weight attribute to label attribute
    if weight is a RandomVariable, then weight is set to the mean

    Args:
        graph: Graph

    Returns:
        Formatted graph
    """
    formatted_graph = graph.copy()
    for node in formatted_graph.nodes:
        formatted_graph.nodes[node]["label"] = formatted_graph.nodes[node]["weight"]
        if isinstance(formatted_graph.nodes[node]["weight"], RandomVariable):
            formatted_graph.nodes[node]["weight"] = formatted_graph.nodes[node]["weight"].mean()
    for edge in formatted_graph.edges:
        formatted_graph.edges[edge]["label"] = formatted_graph.edges[edge]["weight"]
        if isinstance(formatted_graph.edges[edge]["weight"], RandomVariable):
            formatted_graph.edges[edge]["weight"] = formatted_graph.edges[edge]["weight"].mean()
    return formatted_graph

def draw_task_graph(task_graph: nx.DiGraph,
                    axis: Optional[plt.Axes] = None,
                    schedule: Optional[Dict[Hashable, List[Task]]] = None,
                    use_latex: bool = False,
                    node_size: int = 2000,
                    linewidths: int = 2,
                    arrowsize: int = 20,
                    font_size: int = 20,
                    weight_font_size: int = 12,
                    figsize: Tuple[int, int] = None) -> plt.Axes:
    """Draws a task graph

    Args:
        task_graph: Task graph
        axis: Axes to draw on
        schedule: Schedule for coloring nodes
        use_latex: Whether to use latex for labels. Defaults to False.
        node_size: Node size. Defaults to 750.
        linewidths: Line width. Defaults to 2.
        arrowsize: Arrow size. Defaults to 20.
        font_size: Font size. Defaults to 20.
        weight_font_size: Weight font size. Defaults to 12.
        figsize: Figure size. Defaults to None.
    """
    if axis is None:
        # make size slightly larger than default
        _, axis = plt.subplots(figsize=figsize)

    task_graph = format_graph(task_graph.copy())
    pos = nx.nx_agraph.graphviz_layout(task_graph, prog="dot")

    colors, tasks = {}, {}
    if schedule is not None:
        tasks = {task.name: task for node, tasks in schedule.items() for task in tasks}
        network_nodes = set(schedule.keys())

        cmap = plt.get_cmap("tab20", len(network_nodes))
        sorted_nodes = sorted(network_nodes)
        sorted_colors = [cmap(i) for i in range(len(network_nodes))]
        colors = dict(zip(sorted_nodes, sorted_colors))

    nx.draw_networkx_nodes(
        task_graph, pos=pos, ax=axis,
        node_size=1,
        node_color="white",
        edgecolors="black",
        linewidths=linewidths,
    )

    nx.draw_networkx_edges(
        task_graph, pos=pos, ax=axis,
        arrowsize=arrowsize, arrowstyle="->",
        width=linewidths, edge_color="black",
        node_size=node_size,
    )

    for task_name in task_graph.nodes:
        color = "white"
        if schedule is not None and task_name in tasks:
            color = colors[tasks[task_name].node]
        task_label = r"$%s$" % task_name if use_latex else task_name
        nx.draw_networkx_labels(
            task_graph, pos=pos, ax=axis,
            font_size=font_size,
            labels={
                task_name: task_label
            },
            bbox={
                "facecolor": color,
                "edgecolor": "black",
                "boxstyle": "round,pad=0.5",
                # line widths
                "linewidth": linewidths,

            }
        )
        if use_latex:
            cost_label = r"$c(%s)=%s$" % (task_name, round(task_graph.nodes[task_name]['label'], 1))
        else:
            cost_label = f"c({task_name})={round(task_graph.nodes[task_name]['label'], 1)}"
        axis.annotate(
            cost_label,
            xy=pos[task_name],
            xytext=(pos[task_name][0] + font_size//4, pos[task_name][1] - font_size//4),
            fontsize=weight_font_size,
        )

    edge_labels = {}
    for u, v in task_graph.edges:
        label = task_graph.edges[(u, v)]['label']
        if isinstance(label, (int, float)):
            if use_latex:
                label = r"$c\left(%s, %s\right)=%s$" % (u, v, round(label, 1))
            else:
                label = f"c({u}, {v})={round(label, 1)}"
        edge_labels[(u, v)] = label
    nx.draw_networkx_edge_labels(
        task_graph, pos=pos, ax=axis,
        edge_labels=edge_labels,
        font_size=weight_font_size,
    )

    axis.margins(0.1)
    axis.axis("off")
    plt.tight_layout()
    return axis

def draw_network(network: nx.Graph,
                 axis: Optional[plt.Axes] = None,
                 draw_colors: bool = True,
                 use_latex: bool = False,
                 node_size: int = 3000,
                 linewidths: int = 2,
                 font_size: int = 20,
                 weight_font_size: int = 12,
                 figsize: Tuple[int, int] = None) -> plt.Axes:
    """Draws a network

    Args:
        network: Network
        axis: Axes to draw on
        draw_colors: Whether to draw colors. Default is True.
        use_latex: Whether to use latex for labels. Defaults to False.
        node_size: Node size. Defaults to 3000.
        linewidths: Line width. Defaults to 2.
        font_size: Font size. Defaults to 20.
        weight_font_size: Weight font size. Defaults to 12.
        figsize: Figure size. Defaults to None.
    """
    if axis is None:
        _, axis = plt.subplots(figsize=figsize)

    # don't drdaw self loops
    network = format_graph(network.copy())
    network.remove_edges_from(nx.selfloop_edges(network))

    # use same colors as task graph
    sorted_nodes = sorted(network.nodes)
    if draw_colors:
        cmap = plt.get_cmap("tab20", len(network.nodes))
        sorted_colors = [cmap(i) for i in range(len(network.nodes))]
        node_colors = {node: color for node, color in zip(sorted_nodes, sorted_colors)}
        colors = [node_colors[node] for node in sorted_nodes]

    # spring layout
    pos = nx.circular_layout(network)
    # draw network nodes with black border and white fill
    nx.draw_networkx_nodes(
        network, pos=pos, ax=axis,
        nodelist=sorted_nodes,
        node_color=colors if draw_colors else "white",
        edgecolors="black",
        node_size=node_size,
        linewidths=linewidths,
    )

    node_labels = {}
    for node in network.nodes:
        label = network.nodes[node]['label']
        if isinstance(label, (int, float)) and use_latex:
            label = r"$%s$" % node
        node_labels[node] = label
        
        if use_latex:
            weight_label = r"$s(%s)=%s$" % (node, round(network.nodes[node]['weight'], 1))
        else:
            weight_label = f"s({node})={round(network.nodes[node]['weight'], 1)}"
        axis.annotate(
            weight_label,
            xy=pos[node],
            xytext=(pos[node][0]+0.19, pos[node][1]-0.02),
            fontsize=weight_font_size,
        )

    nx.draw_networkx_labels(
        network, pos=pos, ax=axis,
        labels=node_labels,
        font_size=font_size,
    )

    # draw network edges
    nx.draw_networkx_edges(
        network, pos=pos, ax=axis,
        edge_color="black",
        width=linewidths,
    )

    edge_labels = {}
    for u, v in network.edges:
        label = network.edges[(u, v)]['label']
        if isinstance(label, (int, float)):
            if use_latex:
                label = r"$s\left(%s, %s\right)=%s$" % (u, v, round(label, 1))
            else:
                label = f"s({u}, {v})={round(label, 1)}"
        edge_labels[(u, v)] = label

    nx.draw_networkx_edge_labels(
        network, pos=pos, ax=axis,
        edge_labels=edge_labels,
        font_size=weight_font_size,
    )

    axis.margins(0.2)
    axis.axis("off")
    plt.tight_layout()
    return axis

def draw_gantt(schedule: Dict[Hashable, List[Task]],
               use_latex: bool = False,
               font_size: int = 20,
               xmax: float = None,
               axis: Optional[plt.Axes] = None,
               figsize: Tuple[int, int] = (10, 4)) -> plt.Axes:
    """Draws a gantt chart

    Args:
        schedule: Schedule
        use_latex: Whether to use latex for labels. Defaults to False.
        font_size: Font size. Defaults to 20.
        xmax: Maximum x value. Defaults to None.
        axis: Axis to draw on. Defaults to None.
        figsize: Figure size. Defaults to None.

    Returns:
        Gantt chart
    """
    # Remove dummy tasks with near 0 duration
    schedule = {
        node: [task for task in tasks if task.end - task.start > 1e-6]
        for node, tasks in schedule.items()
    }

    makespan = max([0 if not tasks else tasks[-1].end for tasks in schedule.values()])
    if xmax is None:
        xmax = makespan
    else:
        xmax = max(xmax, makespan)
    
    # re-label task and node names to latex
    if use_latex:
        for node in schedule:
            for task in schedule[node]:
                task.node = r"$%s$" % task.node
                task.name = r"$%s$" % task.name

    if use_latex:
        schedule = {r"$%s$" % node: tasks for node, tasks in schedule.items()}

    # insert dummy tasks to make sure all nodes have at least one task
    for node in schedule:
        if len(schedule[node]) == 0:
            schedule[node].append(Task(name=r"", start=0, end=0, node=node))

    data_frame = pd.DataFrame(
        [
            {
                "Task": task.name,
                "Start": task.start,
                "Finish": task.end,
                "Node": task.node
            }
            for _, tasks in schedule.items()
            for task in tasks
        ]
    )
    data_frame['delta'] = data_frame['Finish'] - data_frame['Start']

    # Get the unique set of nodes from the entire DataFrame (including dummy rows)
    unique_nodes = sorted(data_frame['Node'].unique())
    
    # Create a figure and axis
    if axis is None:
        _, axis = plt.subplots(figsize=figsize)
    
    # Plot each task as a horizontal bar with labels
    for index, row in data_frame.iterrows():
        if row['Task'] != '$dummy$':
            # Plot the bar with white color and black border
            axis.barh(
                row['Node'], row['delta'], left=row['Start'],
                color='white', edgecolor='black'
            )
            # Add the task label in the center of the bar
            axis.text(
                row['Start'] + row['delta'] / 2, row['Node'],
                row['Task'], ha='center', va='center', color='black',
                fontsize=font_size,
            )
    
    # Set the y-ticks to be the unique set of nodes
    axis.set_yticks(unique_nodes)
    axis.set_yticklabels(unique_nodes)

    # set tick font size
    for tick in axis.xaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)
    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)
    
    # Set labels and title
    axis.set_xlabel('Time', fontsize=font_size)
    axis.set_ylabel('Nodes', fontsize=font_size)
    axis.set_xlim(0, data_frame['Finish'].max())
    # axis.set_title('Gantt Chart by Node (All Nodes with Task Labels)')
    axis.grid(True, which='both', linestyle='--', linewidth=0.5)
    axis.set_axisbelow(True)

    # set max x limit
    axis.set_xlim(0, xmax)

    # axis.margins(0.2)
    # axis.axis("off")
    plt.tight_layout()
    return axis
