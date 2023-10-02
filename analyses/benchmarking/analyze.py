# Importing required libraries to load and examine the data
import pathlib
from typing import List, Optional, Union, Callable, Iterable
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


thisdir = pathlib.Path(__file__).parent.absolute()

DATASET_ORDER = [
    "in_trees", "out_trees", "chains",

    "blast", "bwa", "cycles", "epigenomics",
    "genome", "montage", "seismology", "soykb",
    "srasearch",

    "etl", "predict", "stats", "train",
]

def load_data() -> pd.DataFrame:
    data = None
    for path in thisdir.glob("results/*.csv"):
        df_dataset = pd.read_csv(path, index_col=0)
        df_dataset["dataset"] = path.stem
        if data is None:
            data = df_dataset
        else:
            data = pd.concat([data, df_dataset], ignore_index=True)
    return data

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
                     xorder: Callable[[Iterable[str]], Iterable[str]] = None,
                     yorder: Callable[[Iterable[str]], Iterable[str]] = None,
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
        ax (plt.Axes, optional): matplotlib axes. Defaults to None.

    Returns:
        plt.Axes: matplotlib axes
    """
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
            gradient = df_color.values.reshape(1, -1)

            im = ax.imshow(
                gradient,
                cmap=cmap,
                aspect='auto',
                extent=[j, j+1, i, i+1],
                vmin=global_min,
                vmax=global_max
            )

            # Add a border with Rectangle (x, y, width, height)
            rect = Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)

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
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', label=color_label if color_label else color)

    # Make upper bound label of colorbar ">{upper_threshold}"
    if upper_threshold < np.inf:
        cbar.ax.set_yticklabels(
            [f'{tick}' for tick in cbar.get_ticks()][:-1]
            + [f'>{upper_threshold}']
        )

    if title:
        plt.title(title)
    plt.xlabel(x_label if x_label else x)
    plt.ylabel(y_label if y_label else y)

    return ax

def main():
    """Analyze the results."""
    data = load_data()
    data["scheduler"] = data["scheduler"].str.replace("Scheduler", "")
    ax = gradient_heatmap(
        data,
        x="scheduler",
        y="dataset",
        color="makespan_ratio",
        cmap="coolwarm",
        upper_threshold=5.0,
        title="Makespan Ratio",
        x_label="Scheduler",
        y_label="Dataset",
        color_label="Makespan Ratio",
    )
    ax.get_figure().savefig(
        thisdir.joinpath("benchmarking.png"),
        dpi=300,
        bbox_inches='tight'
    )

if __name__ == "__main__":
    main()
