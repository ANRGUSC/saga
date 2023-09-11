# Importing required libraries to load and examine the data
import pathlib
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    """Analyze the results."""
    # data = pd.read_csv(thisdir.joinpath("results.csv"), index_col=0)
    data = None
    for path in thisdir.glob("results/*.csv"):
        df_dataset = pd.read_csv(path, index_col=0)
        df_dataset["dataset"] = path.stem
        if data is None:
            data = df_dataset
        else:
            data = pd.concat([data, df_dataset])

    # strip "Scheduler" from scheduler names
    data["scheduler"] = data["scheduler"].str.replace("Scheduler", "")

    # get mean and q1, q3 for each scheduler/dataset
    df_summary = data.groupby(["scheduler", "dataset"]).agg(
        mean_makespan_ratio=("makespan_ratio", "mean"),
        q1_makespan_ratio=("makespan_ratio", lambda x: x.quantile(0.25)),
        q3_makespan_ratio=("makespan_ratio", lambda x: x.quantile(0.75)),
    ).reset_index()
    df_summary.to_csv(thisdir.joinpath("summary.csv"))

    # Create pivot tables for mean, Q1, and Q3
    pivot_mean = df_summary.pivot(index='dataset', columns='scheduler', values='mean_makespan_ratio')
    pivot_q1 = df_summary.pivot(index='dataset', columns='scheduler', values='q1_makespan_ratio')
    pivot_q3 = df_summary.pivot(index='dataset', columns='scheduler', values='q3_makespan_ratio')

    # Calculate global min and max values for the gradient range
    global_min = pivot_q1.min().min()
    global_max = pivot_q3.max().max()

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a colormap based on global min/max values
    cmap = plt.get_cmap('coolwarm')

    # Initialize plot again with rotated x-axis labels
    fig, ax = plt.subplots(figsize=(12, 8))

    # Loop through each cell in the pivot table to plot its custom gradient
    for i, (dataset, row) in enumerate(pivot_mean.iterrows()):
        for j, (scheduler, mean_value) in enumerate(row.items()):
            # Skip NaN values
            if np.isnan(mean_value):
                continue

            q1_value = pivot_q1.loc[dataset, scheduler]
            q3_value = pivot_q3.loc[dataset, scheduler]

            # Create gradient and plot it as an image
            gradient = np.linspace(q1_value, q3_value, 256).reshape(1, -1)
            im = ax.imshow(gradient, cmap=cmap, aspect='auto', extent=[j, j+1, i, i+1], vmin=global_min, vmax=global_max)

            # Add a border with Rectangle (x, y, width, height)
            rect = Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)


    # Add labels, ticks, and other plot elements
    ax.set_xticks(np.arange(len(pivot_mean.columns)))
    ax.set_yticks(np.arange(len(pivot_mean.index)))
    ax.set_xticklabels(pivot_mean.columns, rotation=90)  # Rotate x-axis labels
    ax.set_yticklabels(pivot_mean.index)

    # Add colorbar
    plt.colorbar(im, ax=ax, orientation='vertical', label='Makespan Ratio')

    plt.title('Mean, Q1, and Q3 Makespan Ratios')
    plt.xlabel('Scheduler')
    plt.ylabel('Dataset')

    plt.savefig(
        thisdir.joinpath("benchmarking.png"),
        dpi=300,
        bbox_inches='tight'
    )





if __name__ == "__main__":
    main()
