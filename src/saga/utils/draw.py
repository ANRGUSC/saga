from functools import lru_cache
import logging
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, TypeVar

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import cm
import matplotlib.colors
from matplotlib.patches import Rectangle
from matplotlib import rc_context
import networkx as nx
import pandas as pd
import numpy as np

from saga import ScheduledTask

# create logger with SAGA:saga.utils.draw: prefix
logger = logging.getLogger("SAGA:saga.utils.draw")

TGraphType = TypeVar("TGraphType", bound=Union[nx.DiGraph, nx.Graph])


def format_graph(graph: TGraphType) -> TGraphType:
    """Formats the graph

    copies weight attribute to label attribute
    if weight is a RandomVariable, then weight is set to the mean

    Args:
        graph: Graph

    Returns:
        Formatted graph
    """
    graph = graph.copy()
    for node in graph.nodes:
        graph.nodes[node]["weight"] = round(graph.nodes[node]["weight"], 2)
    for edge in graph.edges:
        graph.edges[edge]["weight"] = round(graph.edges[edge]["weight"], 2)

    return graph


@lru_cache(maxsize=None)
def is_latex_installed() -> bool:
    return shutil.which("latex") is not None


def draw_task_graph(
    task_graph: nx.DiGraph,
    axis: Optional[Axes] = None,
    schedule: Optional[Dict[str, List["ScheduledTask"]]] = None,
    use_latex: bool = False,
    node_size: int = 2000,
    linewidths: int = 2,
    arrowsize: int = 20,
    font_size: int = 20,
    weight_font_size: int = 12,
    figsize: Optional[Tuple[int, int]] = None,
    draw_node_labels: bool = True,
    draw_edge_weights: bool = True,
    draw_node_weights: bool = True,
    node_weight_offset: float = 0.1,
    pos: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Axes:
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
        draw_node_labels: Whether to draw node labels. Defaults to True.
        draw_edge_weights: Whether to draw edge weights. Defaults to True.
        draw_node_weights: Whether to draw node weights. Defaults to True.
        node_weight_offset: Offset for node weight labels. Defaults to 0.2.
        pos: Position of nodes. Defaults to None.
    """
    if use_latex and not is_latex_installed():
        logger.warning("Latex is not installed. Using non-latex mode.")
        use_latex = False

    rc_context_opts = {"text.usetex": use_latex}
    with rc_context(rc=rc_context_opts):
        if axis is None:
            # make size slightly larger than default
            _, axis = plt.subplots(figsize=figsize)
        if axis is None:
            raise ValueError("Axis could not be created.")

        task_graph = format_graph(task_graph.copy())

        if pos is None:
            pos = nx.nx_agraph.graphviz_layout(task_graph, prog="dot")

        colors: Dict[str, Tuple[float, float, float, float]] = {}
        tasks: Dict[str, ScheduledTask] = {}
        if schedule is not None:
            tasks = {
                task.name: task for node, tasks in schedule.items() for task in tasks
            }
            network_nodes: Set[str] = set(schedule.keys())

            cmap = plt.get_cmap("tab20", len(network_nodes))
            sorted_nodes = sorted(
                map(str, network_nodes)
            )  # sort as strings for consistent coloring
            sorted_colors = [cmap(i) for i in range(len(network_nodes))]
            colors = dict(zip(sorted_nodes, sorted_colors))
        # if nodes have "color" attribute, use that
        elif any("color" in task_graph.nodes[node] for node in task_graph.nodes):
            categories = sorted(
                set(
                    task_graph.nodes[node].get("color", "__blank__")
                    for node in task_graph.nodes
                )
            )
            cmap = plt.get_cmap("tab20", len(categories))
            colors = {
                node: cmap(
                    categories.index(task_graph.nodes[node].get("color", "__blank__"))
                )
                for node in task_graph.nodes
            }

        nx.draw_networkx_nodes(
            task_graph,
            pos=pos,
            ax=axis,
            node_size=1 if draw_node_labels else node_size,
            node_color=[colors.get(str(node), "white") for node in task_graph.nodes],
            edgecolors="black",
            linewidths=linewidths,
        )

        nx.draw_networkx_edges(
            task_graph,
            pos=pos,
            ax=axis,
            arrowsize=arrowsize,
            arrowstyle="->",
            width=linewidths,
            edge_color="black",
            node_size=node_size,
        )

        for task_name in task_graph.nodes:
            if draw_node_labels:
                color: str | Tuple[float, float, float, float] = "white"
                if schedule is not None and task_name in tasks:
                    color = colors[str(tasks[task_name].node)]
                elif task_name in colors:
                    color = colors[str(task_name)]
                task_label = r"$%s$" % task_name if use_latex else task_name
                nx.draw_networkx_labels(
                    task_graph,
                    pos=pos,
                    ax=axis,
                    font_size=font_size,
                    labels={task_name: task_label},
                    bbox={
                        "facecolor": color,
                        "edgecolor": "black",
                        "boxstyle": "round,pad=0.5",
                        # line widths
                        "linewidth": linewidths,
                    },
                )
            if draw_node_weights:
                if use_latex:
                    # if has "label" attribute, use that
                    if "label" in task_graph.nodes[task_name]:
                        cost_label = r"$%s$" % (task_graph.nodes[task_name]["label"])
                    else:
                        cost_label = r"$%s$" % (
                            round(task_graph.nodes[task_name]["weight"], 2)
                        )
                else:
                    cost_label = f"{round(task_graph.nodes[task_name]['weight'], 2)}"

                axis.annotate(
                    cost_label,
                    xy=pos[task_name],
                    xytext=(font_size, 0),
                    textcoords="offset points",
                    fontsize=weight_font_size,
                )

        if draw_edge_weights:
            edge_labels = {}
            for u, v in task_graph.edges:
                label = task_graph.edges[(u, v)]["weight"]
                if isinstance(label, (int, float)):
                    if use_latex:
                        if "label" in task_graph.edges[(u, v)]:
                            label = r"$%s$" % (task_graph.edges[(u, v)]["label"])
                        else:
                            label = r"$%s$" % (round(label, 2))
                    else:
                        label = f"{round(label, 2)}"
                edge_labels[(u, v)] = label
            nx.draw_networkx_edge_labels(
                task_graph,
                pos=pos,
                ax=axis,
                edge_labels=edge_labels,
                font_size=weight_font_size,
                rotate=False,
            )

        axis.margins(0.1)
        axis.axis("off")
        # plt.tight_layout()
        return axis


def draw_network(
    network: nx.Graph,
    axis: Optional[Axes] = None,
    draw_colors: bool = True,
    use_latex: bool = False,
    node_size: int = 3000,
    linewidths: int = 2,
    font_size: int = 20,
    weight_font_size: int = 12,
    figsize: Optional[Tuple[int, int]] = None,
    draw_node_labels: bool = True,
    draw_edge_weights: bool = True,
    draw_node_weights: bool = True,
) -> Axes:
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
        draw_node_labels: Whether to draw node labels. Defaults to True.
        draw_edge_weights: Whether to draw edge weights. Defaults to True.
        draw_node_weights: Whether to draw node weights. Defaults to True.
    """
    if use_latex and not is_latex_installed():
        logger.warning("Latex is not installed. Using non-latex mode.")
        use_latex = False

    rc_context_opts = {"text.usetex": use_latex}
    with rc_context(rc=rc_context_opts):
        if axis is None:
            _, axis = plt.subplots(figsize=figsize)
            if axis is None:
                raise ValueError("Axis could not be created.")

        # don't draw self loops
        network = format_graph(network.copy())
        network.remove_edges_from(nx.selfloop_edges(network))

        # use same colors as task graph
        sorted_nodes = sorted(network.nodes)
        colors: List[str] | List[Tuple[float, float, float, float]] = ["white"] * len(
            sorted_nodes
        )
        if draw_colors:
            cmap = plt.get_cmap("tab20", len(network.nodes))
            sorted_colors = [cmap(i) for i in range(len(network.nodes))]
            node_colors = {
                node: color for node, color in zip(sorted_nodes, sorted_colors)
            }
            colors = [node_colors[node] for node in sorted_nodes]

        # spring layout
        pos = nx.circular_layout(network)
        # draw network nodes with black border and white fill
        nx.draw_networkx_nodes(
            network,
            pos=pos,
            ax=axis,
            nodelist=sorted_nodes,
            node_color=colors,
            edgecolors="black",
            node_size=node_size,
            linewidths=linewidths,
        )

        node_labels = {}
        for node in network.nodes:
            if draw_node_labels:
                label = network.nodes[node].get("label", node)
                # if isinstance(label, (int, float)) and use_latex:
                if use_latex:
                    label = r"$%s$" % node
                node_labels[node] = label
            if draw_node_weights:
                if use_latex:
                    weight_label = r"$%s$" % (round(network.nodes[node]["weight"], 2))
                else:
                    weight_label = f"{round(network.nodes[node]['weight'], 2)}"

                axis.annotate(
                    weight_label,
                    xy=(pos[node][0], pos[node][1]),
                    xytext=(font_size * 1.5, 0),
                    textcoords="offset points",
                    fontsize=weight_font_size,
                )

        if draw_node_labels:
            nx.draw_networkx_labels(
                network,
                pos=pos,
                ax=axis,
                labels=node_labels,
                font_size=font_size,
            )

        # draw network edges
        nx.draw_networkx_edges(
            network,
            pos=pos,
            ax=axis,
            edge_color="black",
            width=linewidths,
        )

        if draw_edge_weights:
            edge_labels = {}
            for u, v in network.edges:
                label = network.edges[(u, v)].get(
                    "label", network.edges[(u, v)]["weight"]
                )
                if isinstance(label, (int, float)):
                    if use_latex:
                        label = r"$%s$" % (round(label, 2))
                    else:
                        label = f"{round(label, 2)}"
                edge_labels[(u, v)] = label

            nx.draw_networkx_edge_labels(
                network,
                pos=pos,
                ax=axis,
                edge_labels=edge_labels,
                font_size=weight_font_size,
                rotate=False,
            )

        axis.margins(0.2)
        axis.axis("off")
        plt.tight_layout()
        return axis


def draw_gantt(
    schedule: Dict[str, List["ScheduledTask"]],
    use_latex: bool = False,
    font_size: int = 20,
    tick_font_size: int = 20,
    xmax: Optional[float] = None,
    axis: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 4),
    draw_task_labels: bool = True,
) -> Axes:
    """Draws a gantt chart

    Args:
        schedule: Schedule
        use_latex: Whether to use latex for labels. Defaults to False.
        font_size: Font size. Defaults to 20.
        xmax: Maximum x value. Defaults to None.
        axis: Axis to draw on. Defaults to None.
        figsize: Figure size. Defaults to None.
        draw_task_labels: Whether to draw task labels. Defaults to True.

    Returns:
        Gantt chart
    """
    if use_latex and not is_latex_installed():
        logger.warning("Latex is not installed. Using non-latex mode.")
        use_latex = False

    rc_context_opts = {"text.usetex": use_latex}
    with rc_context(rc=rc_context_opts):
        # Remove dummy tasks with near 0 duration
        schedule = {
            node: [task for task in tasks if task.end - task.start > 1e-6]
            for node, tasks in schedule.items()
        }

        makespan = max(
            [0 if not tasks else tasks[-1].end for tasks in schedule.values()]
        )
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
                schedule[node].append(
                    ScheduledTask(name=r"", start=0, end=0, node=node)
                )

        data_frame = pd.DataFrame(
            [
                {
                    "Task": task.name,
                    "Start": task.start,
                    "Finish": task.end,
                    "Node": task.node,
                }
                for _, tasks in schedule.items()
                for task in tasks
            ]
        )
        data_frame["delta"] = data_frame["Finish"] - data_frame["Start"]

        # Get the unique set of nodes from the entire DataFrame (including dummy rows)
        unique_nodes = sorted(data_frame["Node"].unique())

        # Create a figure and axis
        if axis is None:
            _, axis = plt.subplots(figsize=figsize)
            if axis is None:
                raise ValueError("Axis could not be created.")

        # Plot each task as a horizontal bar with labels
        for index, row in data_frame.iterrows():
            if row["Task"] != "$dummy$":
                # Plot the bar with white color and black border
                axis.barh(
                    row["Node"],
                    row["delta"],
                    left=row["Start"],
                    color="white",
                    edgecolor="black",
                )
                # Add the task label in the center of the bar
                if draw_task_labels:
                    axis.text(
                        row["Start"] + row["delta"] / 2,
                        row["Node"],
                        row["Task"],
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=font_size,
                    )

        # Set the y-ticks to be the unique set of nodes
        axis.set_yticks(unique_nodes)
        axis.set_yticklabels(unique_nodes)

        # set tick font size
        axis.tick_params(axis="both", which="major", labelsize=tick_font_size)

        # Set labels and title
        axis.set_xlabel("Time", fontsize=font_size)
        axis.set_ylabel("Nodes", fontsize=font_size)
        axis.set_xlim(0, data_frame["Finish"].max())
        # axis.set_title('Gantt Chart by Node (All Nodes with Task Labels)')
        axis.grid(True, which="both", linestyle="--", linewidth=0.5)
        axis.set_axisbelow(True)

        # set max x limit
        axis.set_xlim(0, xmax)

        # axis.margins(0.2)
        # axis.axis("off")
        plt.tight_layout()
        return axis


def gradient_heatmap(
    data: pd.DataFrame,
    x: Union[str, List[str]],
    y: Union[str, List[str]],
    color: str,
    cmap: str = "coolwarm",
    upper_threshold: float = np.inf,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    rotate_xlabels: int = 90,
    y_label: Optional[str] = None,
    color_label: Optional[str] = None,
    cell_text: Optional[str] = None,
    cell_font_size: Optional[float] = None,
    xorder: Optional[Callable[[Any], Any]] = None,
    yorder: Optional[Callable[[Any], Any]] = None,
    ax: Optional[Axes] = None,
    font_size: float = 20.0,
    figsize: Tuple[float, float] = (12, 8),
    linewidth: int = 1,
    cmap_lower: float = 0.0,
    cmap_upper: float = 1.0,
    use_latex: bool = False,
) -> Axes:
    """Create a heatmap with a custom gradient for each cell.

    Args:
        data (pd.DataFrame): data to plot
        x (str): column name for x-axis
        y (str): column name for y-axis
        color (str): column name for color
        cmap (str, optional): matplotlib colormap. Defaults to "coolwarm".
        upper_threshold (Optional[float], optional): upper bound for colorbar. Defaults to None.
        title (str, optional): plot title. Defaults to None.
        x_label (str, optional): x-axis label. Defaults to None.
        y_label (str, optional): y-axis label. Defaults to None.
        color_label (str, optional): colorbar label. Defaults to None.
        cell_text (str, optional): indicates the column name to use for cell labels. None indicates no label. Default is None.
        cell_font_size (float, optional): font size for cell labels. None indicates no label. Default is None.
        xorder (Callable[[Any], Any], optional): sort key function to order x-axis values. Defaults to None.
        yorder (Callable[[Any], Any], optional): sort key function to order y-axis values. Defaults to None.
        ax (Axes, optional): matplotlib axes. Defaults to None.
        font_size (float, optional): font size. Defaults to 20.0.
        figsize (Tuple[float, float], optional): figure size. Defaults to (12, 8).
        linewidth (int, optional): linewidth for cell borders. Defaults to 1.
        cmap_lower (float, optional): lower bound for colormap. Defaults to 0.0.
        cmap_upper (float, optional): upper bound for colormap. Defaults to 1.0.
        use_latex (bool, optional): use LaTeX for text rendering. Defaults to False.

    Returns:
        Axes: matplotlib axes
    """
    if use_latex and not is_latex_installed():
        logger.warning("LaTeX is not installed. Falling back to non-LaTeX rendering.")
        use_latex = False

    rc_context_opts = {"text.usetex": use_latex, "font.size": font_size}
    with rc_context(rc=rc_context_opts):
        data = data.copy()
        # combine xs and ys into a single column if necessary
        # make column categorical and sorted by x/y order
        if isinstance(x, list):
            col_name = "/".join(x)
            data[col_name] = data[x].apply(
                lambda row: "/".join(row.values.astype(str)), axis=1
            )
            categories: List[str] | Callable[[Any], Any] = [
                "/".join(map(str, row))
                for row in sorted(
                    map(str, data[x].drop_duplicates().itertuples(index=False)),
                    key=xorder,
                )
            ]
            categories = xorder if xorder else categories
            data[col_name] = pd.Categorical(
                data[col_name], categories=categories, ordered=True
            )
            x = col_name
        else:
            categories = sorted(data[x].drop_duplicates(), key=xorder)
            data[x] = pd.Categorical(data[x], categories=categories, ordered=True)

        if isinstance(y, list):
            col_name = "/".join(y)
            data[col_name] = data[y].apply(
                lambda row: "/".join(row.values.astype(str)), axis=1
            )
            categories = [
                "/".join(map(str, row))
                for row in sorted(
                    map(str, data[y].drop_duplicates().itertuples(index=False)),
                    key=yorder,
                )
            ]
            data[col_name] = pd.Categorical(
                data[col_name], categories=categories, ordered=True
            )
            y = col_name
        else:
            categories = sorted(data[y].drop_duplicates(), key=yorder)
            data[y] = pd.Categorical(data[y], categories=categories, ordered=True)

        global_min = data[color].min()
        global_max = min(data[color].max(), upper_threshold)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            if ax is None:
                raise ValueError("Axis could not be created.")

        _cmap = cm.get_cmap(cmap)
        _cmap_arr = _cmap(np.linspace(cmap_lower, cmap_upper, _cmap.N))
        listed_cmap = matplotlib.colors.ListedColormap(_cmap_arr)

        # Get unique values for x and y in the correct order (by category)
        xvals = data[x].drop_duplicates().sort_values()
        yvals = data[y].drop_duplicates().sort_values(ascending=False)

        # Initialize im to None - it will be set when we have data to plot
        im = None

        for i, yval in enumerate(yvals):
            for j, xval in enumerate(xvals):
                df_color = data.loc[(data[x] == xval) & (data[y] == yval), color]
                if isinstance(df_color, pd.Series):
                    df_color = df_color.sort_values()
                else:
                    df_color = pd.Series([df_color])
                if df_color.empty:  # add a white cell if there is no data
                    rect = Rectangle(
                        (j, i),
                        1,
                        1,
                        linewidth=linewidth,
                        edgecolor="black",
                        facecolor="white",
                    )
                else:
                    gradient = np.asarray(df_color.values).reshape(1, -1)

                    im = ax.imshow(
                        gradient,
                        cmap=listed_cmap,
                        aspect="auto",
                        extent=(float(j), float(j + 1), float(i), float(i + 1)),
                        vmin=global_min,
                        vmax=global_max,
                    )
                    rect = Rectangle(
                        (j, i),
                        1,
                        1,
                        linewidth=linewidth,
                        edgecolor="black",
                        facecolor="none",
                    )

                ax.add_patch(rect)

                if not df_color.empty and cell_font_size is not None:
                    value = df_color.mean()
                    if np.isnan(value):
                        value = ""
                    elif value > 1000:
                        value = f"$>{1000}$" if use_latex else ">1000"
                    elif value > upper_threshold:
                        value = (
                            f"$>{upper_threshold}$"
                            if use_latex
                            else f">{upper_threshold}"
                        )
                    elif isinstance(value, int) or (
                        isinstance(value, float) and value.is_integer()
                    ):
                        value = f"${int(value)}$" if use_latex else f"{int(value)}"
                    else:
                        value = f"${value:.2f}$" if use_latex else f"{value:.2f}"
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        value,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=cell_font_size,
                    )

        # Add labels, ticks, and other plot elements
        ax.set_xticks(np.arange(len(xvals)) + 0.5)  # Adjusted tick positions
        ax.set_yticks(np.arange(len(yvals)) + 0.5)  # Adjusted tick positions
        ax.set_xticklabels(xvals, rotation=rotate_xlabels, fontsize=font_size)
        ax.set_yticklabels(yvals, fontsize=font_size)
        ax.grid(False)

        ax.tick_params(
            axis="x", which="both", bottom=False, top=False
        )  # Optional: remove bottom ticks
        ax.tick_params(
            axis="y", which="both", left=False, right=False
        )  # Optional: remove left ticks

        ax.set_xlim(0, len(xvals))  # Set x-axis limits
        ax.set_ylim(0, len(yvals))  # Set y-axis limits

        # Add colorbar if we have data to plot
        if im is not None:
            fig = ax.get_figure()
            if fig is not None:
                cbar = fig.colorbar(
                    im,
                    ax=ax,
                    orientation="vertical",
                    label=color_label if color_label else color,
                )

                # Make upper bound label of colorbar ">{upper_threshold}"
                if upper_threshold < np.inf:
                    cbar.ax.set_yticklabels(
                        [f"{tick:0.2f}" for tick in cbar.get_ticks()][:-1]
                        + [
                            f"$> {upper_threshold}$"
                            if use_latex
                            else f"> {upper_threshold}"
                        ],
                        fontsize=font_size,
                    )

        if title:
            ax.set_title(title)
        ax.set_xlabel(x_label if x_label else x, labelpad=20, fontsize=font_size)
        ax.set_ylabel(y_label if y_label else y, labelpad=20, fontsize=font_size)

        return ax
