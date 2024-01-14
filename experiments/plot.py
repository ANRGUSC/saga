from typing import Callable, Iterable, List, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def gradient_heatmap(data: pd.DataFrame,
                     x: Union[str, List[str]], # pylint: disable=invalid-name
                     y: Union[str, List[str]], # pylint: disable=invalid-name
                     color: str, # pylint: disable=invalid-name
                     cmap: str = "coolwarm",
                     upper_threshold: float = np.inf,
                     title: str = None,
                     x_label: str = None,
                     y_label: str = None,
                     color_label: str = None,
                     include_cell_labels: bool = False,
                     xorder: Callable[[Union[str, Iterable[str]]], Union[str, Iterable[str]]] = None,
                     yorder: Callable[[Union[str, Iterable[str]]], Union[str, Iterable[str]]] = None,
                     ax: plt.Axes = None) -> plt.Axes: # pylint: disable=invalid-name
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
        include_cell_labels (bool, optional): whether to include text in each cell. Defaults to False.
        xorder (Callable[[Union[str, Iterable[str]]], Union[str, Iterable[str]]], optional): function to order x-axis. Defaults to None.
        yorder (Callable[[Union[str, Iterable[str]]], Union[str, Iterable[str]]], optional): function to order y-axis. Defaults to None.
        ax (plt.Axes, optional): matplotlib axes. Defaults to None.

    Returns:
        plt.Axes: matplotlib axes
    """
    plt.rcParams.update({
        'font.size': 20,
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'text.usetex': True,
    })

    data = data.copy()
    # combine xs and ys into a single column if necessary
    # make column categorical and sorted by x/y order
    if isinstance(x, list):
        col_name = "/".join(x)
        data[col_name] = data[x].apply(lambda row: "/".join(row.values.astype(str)), axis=1)
        categories = [
            "/".join(map(str, row))
            for row in sorted(
                data[x].drop_duplicates().itertuples(index=False),
                key=xorder
            )
        ]
        categories = xorder if xorder else categories
        data[col_name] = pd.Categorical(data[col_name], categories=categories, ordered=True)
        x = col_name
    else:
        categories = sorted(data[x].drop_duplicates(), key=xorder)
        data[x] = pd.Categorical(data[x], categories=categories, ordered=True)

    if isinstance(y, list):
        col_name = "/".join(y)
        data[col_name] = data[y].apply(lambda row: "/".join(row.values.astype(str)), axis=1)
        categories = [
            "/".join(map(str, row))
            for row in sorted(
                data[y].drop_duplicates().itertuples(index=False),
                key=yorder
            )
        ]
        data[col_name] = pd.Categorical(data[col_name], categories=categories, ordered=True)
        y = col_name
    else:
        categories = sorted(data[y].drop_duplicates(), key=yorder)
        data[y] = pd.Categorical(data[y], categories=categories, ordered=True)

    global_min = data[color].min()
    global_max = min(data[color].max(), upper_threshold)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))

    # Get unique values for x and y in the correct order (by category)
    xvals = data[x].drop_duplicates().sort_values()
    yvals = data[y].drop_duplicates().sort_values(ascending=False)
    for i, yval in enumerate(yvals):
        for j, xval in enumerate(xvals):
            df_color = data.loc[(data[x] == xval) & (data[y] == yval), color].sort_values()
            if df_color.empty: # add a white cell if there is no data
                rect = Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='white')
            else:
                gradient = df_color.values.reshape(1, -1)

                im = ax.imshow(
                    gradient,
                    cmap=cmap,
                    aspect='auto',
                    extent=[j, j+1, i, i+1],
                    vmin=global_min,
                    vmax=global_max
                )
                rect = Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
            
            ax.add_patch(rect)

            if not df_color.empty and include_cell_labels:
                value = df_color.mean()
                if np.isnan(value):
                    value = ""
                elif value > 1000:
                    value = f'$>{1000}$'
                elif value > upper_threshold:
                    value = f'$>{upper_threshold}$'
                else:
                    value = f'${value:.2f}$'
                ax.text(
                    j+0.5, i+0.5, value,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=10
                )

    # Add labels, ticks, and other plot elements
    ax.set_xticks(np.arange(len(xvals)) + 0.5)  # Adjusted tick positions
    ax.set_yticks(np.arange(len(yvals)) + 0.5)    # Adjusted tick positions
    ax.set_xticklabels(xvals, rotation=90)      # Rotate x-axis labels
    ax.set_yticklabels(yvals)

    ax.tick_params(axis='x', which='both', bottom=False, top=False)  # Optional: remove bottom ticks
    ax.tick_params(axis='y', which='both', left=False, right=False)  # Optional: remove left ticks

    ax.set_xlim([0, len(xvals)])  # Set x-axis limits
    ax.set_ylim([0, len(yvals)])  # Set y-axis limits

    # Add colorbar
    cbar = plt.colorbar(
        im, ax=ax,
        orientation='vertical',
        label=color_label if color_label else color
    )

    # Make upper bound label of colorbar ">{upper_threshold}"
    if upper_threshold < np.inf:
        cbar.ax.set_yticklabels(
            [f'{tick:0.2f}' for tick in cbar.get_ticks()][:-1]
            + [f'$> {upper_threshold}$']
        )

    if title:
        plt.title(title)
    plt.xlabel(x_label if x_label else x, labelpad=20)
    plt.ylabel(y_label if y_label else y, labelpad=20)

    return ax