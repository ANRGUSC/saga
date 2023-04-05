import logging
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
from typing import List, Optional, Dict, Hashable
from ..base import Task, Scheduler


def draw_task_graph(task_graph: nx.DiGraph, 
                    ax: Optional[plt.Axes] = None,
                    schedule: Optional[Dict[Hashable, List[Task]]] = None) -> plt.Axes:
    """Draws a task graph

    Args:
        task_graph: Task graph
        ax: Axes to draw on
        schedule: Schedule for coloring nodes
    """
    if ax is None:
        fig, ax = plt.subplots()
    pos = nx.nx_agraph.graphviz_layout(task_graph, prog="dot")

    colors = {}
    if schedule is not None:
        tasks = {task.name: task for node, tasks in schedule.items() for task in tasks}
        network_nodes = {task.node for node, tasks in schedule.items() for task in tasks}
        network_node_idx = {node: i for i, node in enumerate(sorted(network_nodes))}
        cmap = plt.get_cmap("tab20", len(network_nodes))
        colors = {node: cmap(idx) for node, idx in network_node_idx.items()}

    nx.draw_networkx_edges(
        task_graph, pos=pos, ax=ax, 
        arrowsize=20, arrowstyle="->", 
        width=2, edge_color="black",
        node_size=750,
    )

    for task_name in task_graph.nodes:
        color = "white"
        try:
            color = colors[tasks[task_name].node]
        except Exception:
            logging.warning(f"Could not get color for {task_name}")
            
        print(f"task_name: {task_name} color: {color}")
        nx.draw_networkx_labels(
            task_graph, pos=pos, ax=ax,
            labels={
                task_name: f"{task_name} ({task_graph.nodes[task_name]['weight']:.2f})"
            },
            bbox=dict(
                facecolor=color,
                edgecolor="black",
                boxstyle="round,pad=0.5"
            )
        )
    nx.draw_networkx_edge_labels(
        task_graph, pos=pos, ax=ax,
        edge_labels={
            (u, v): f"{task_graph.edges[(u, v)]['weight']:.2f}"
            for u, v in task_graph.edges
        }
    )

    return ax


def draw_network(network: nx.Graph, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Draws a network

    Args:
        network: Network
        ax: Axes to draw on
    """
    if ax is None:
        fig, ax = plt.subplots()

    # don't drdaw self loops
    network = network.copy()
    network.remove_edges_from(nx.selfloop_edges(network))

    # use same colors as task graph
    cmap = plt.get_cmap("tab20", len(network.nodes))
    node_idx = {node: i for i, node in enumerate(sorted(network.nodes))}
    colors = [cmap(idx) for node, idx in node_idx.items()]
    
    # spring layout
    pos = nx.spring_layout(network)
    # draw network nodes with black border and white fill
    nx.draw_networkx_nodes(
        network, pos=pos, ax=ax,
        node_color=colors,
        edgecolors="black",
        node_size=3000
    )

    # draw network edges
    nx.draw_networkx_edges(
        network, pos=pos, ax=ax,
        edge_color="black",
    )


    # draw "weight" labels on nodes and edges
    nx.draw_networkx_labels(
        network, pos=pos, ax=ax,
        labels={
            node: f"{node} ({network.nodes[node]['weight']:.2f})"
            for node in network.nodes
        }
    )
    nx.draw_networkx_edge_labels(
        network, pos=pos, ax=ax, 
        edge_labels={
            (u, v): f"{network.edges[(u, v)]['weight']:.2f}"
            for u, v in network.edges
        }
    )
    return ax
    

def draw_gantt(schedule: Dict[Hashable, Task]) -> Figure:
    """Draws a Gantt chart

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

    df = pd.DataFrame(
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
    df['delta'] = df['Finish'] - df['Start']

    fig = px.timeline(
        df, 
        title="Schedule",
        x_start="Start", 
        x_end="Finish", 
        y="Node", 
        text="Task",
        template="plotly_white"
    )
    fig.layout.xaxis.type = "linear"
    fig.data[0].x = df.delta.tolist()

    # set x-axis label to "Time"
    fig.update_layout(xaxis_title="Time")

    # center title
    fig.update_layout(title_x=0.5)

    # give bars outline
    fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=1)

    fig.update_yaxes(range=[-1/2, len(schedule)+1/2])
    return fig