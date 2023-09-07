import logging
from typing import Dict, Hashable, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

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
                    schedule: Optional[Dict[Hashable, List[Task]]] = None) -> plt.Axes:
    """Draws a task graph

    Args:
        task_graph: Task graph
        axis: Axes to draw on
        schedule: Schedule for coloring nodes
    """
    if axis is None:
        _, axis = plt.subplots()

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
        node_size=100,
        node_color="white",
        edgecolors="black",
        linewidths=2,
    )

    nx.draw_networkx_edges(
        task_graph, pos=pos, ax=axis,
        arrowsize=20, arrowstyle="->",
        width=2, edge_color="black",
        node_size=750,
    )

    for task_name in task_graph.nodes:
        color = "white"
        try:
            color = colors[tasks[task_name].node]
        except KeyError:
            if schedule is not None:
                logging.warning("Could not get color for %s", task_name)

        nx.draw_networkx_labels(
            task_graph, pos=pos, ax=axis,
            labels={
                task_name: f"{task_name} ({task_graph.nodes[task_name]['label']:.2f})"
            },
            bbox={
                "facecolor": color,
                "edgecolor": "black",
                "boxstyle": "round,pad=0.5"
            }
        )
    nx.draw_networkx_edge_labels(
        task_graph, pos=pos, ax=axis,
        edge_labels={
            (u, v): f"{task_graph.edges[(u, v)]['label']:.2f}"
            for u, v in task_graph.edges
        }
    )

    axis.margins(0.1)
    axis.axis("off")
    plt.tight_layout()
    return axis

def draw_network(network: nx.Graph,
                 axis: Optional[plt.Axes] = None,
                 draw_colors: bool = True) -> plt.Axes:
    """Draws a network

    Args:
        network: Network
        axis: Axes to draw on
        draw_colors: Whether to draw colors. Default is True.
    """
    if axis is None:
        _, axis = plt.subplots()

    # don't drdaw self loops
    network = format_graph(network.copy())
    network.remove_edges_from(nx.selfloop_edges(network))

    # use same colors as task graph
    cmap = plt.get_cmap("tab20", len(network.nodes))
    sorted_nodes = sorted(network.nodes)
    sorted_colors = [cmap(i) for i in range(len(network.nodes))]
    node_colors = {node: color for node, color in zip(sorted_nodes, sorted_colors)}
    colors = [node_colors[node] for node in sorted_nodes]

    # spring layout
    pos = nx.spring_layout(network)
    # draw network nodes with black border and white fill
    nx.draw_networkx_nodes(
        network, pos=pos, ax=axis,
        nodelist=sorted_nodes,
        node_color=colors if draw_colors else "white",
        edgecolors="black",
        node_size=3000
    )

    # draw network edges
    nx.draw_networkx_edges(
        network, pos=pos, ax=axis,
        edge_color="black",
    )

    nx.draw_networkx_labels(
        network, pos=pos, ax=axis,
        labels={
            node: f"{node} ({network.nodes[node]['label']:.2f})"
            for node in network.nodes
        }
    )
    nx.draw_networkx_edge_labels(
        network, pos=pos, ax=axis,
        edge_labels={
            (u, v): f"{network.edges[(u, v)]['label']:.2f}"
            for u, v in network.edges
        }
    )

    axis.margins(0.1)
    axis.axis("off")
    plt.tight_layout()
    return axis


def draw_gantt(schedule: Dict[Hashable, List[Task]]) -> Figure:
    """Draws a gantt chart

    Args:
        schedule: Schedule

    Returns:
        Gantt chart
    """
    # Remove dummy tasks with near 0 duration
    schedule = {
        node: [task for task in tasks if task.end - task.start > 1e-6]
        for node, tasks in schedule.items()
    }

    makespan = max(task.end for tasks in schedule.values() for task in tasks)

    # insert dummy tasks to make sure all nodes have at least one task
    for node in schedule:
        if len(schedule[node]) == 0:
            schedule[node].append(Task(name="dummy", start=-1, end=-1, node=node))

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

    fig = px.timeline(
        data_frame,
        x_start="Start",
        x_end="Finish",
        y="Node",
        text="Task",
        template="plotly_white"
    )
    fig.layout.xaxis.type = "linear"
    fig.data[0].x = data_frame.delta.tolist()

    # set x-axis label to "Time"
    fig.update_layout(xaxis_title="Time")

    # make the bars black with black outline
    fig.update_traces(
        marker_color='white',
        marker_line_color='black',
        marker_line_width=1.5,
        opacity=1
    )

    # Integer labels on y-axis
    min_node = min(schedule.keys())
    max_node = max(schedule.keys())
    fig.update_yaxes(tickvals=list(range(min_node, max_node+1)))
    fig.update_yaxes(dtick=1)

    # set y-axis range from 0 to len(schedule) + 1
    fig.update_yaxes(range=[min_node-1/2, max_node+1/2])
    fig.update_xaxes(range=[0, makespan+0.1])

    # set aspect ratio high-res 2:1
    fig.update_layout(width=1200, height=600)

    # make font larger
    fig.update_layout(font_size=20)

    # remove margins
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig
