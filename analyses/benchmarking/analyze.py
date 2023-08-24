# Importing required libraries to load and examine the data
import json
import pathlib

import pandas as pd
import plotly.express as px

thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    """Analyze the results."""
    data = json.loads(thisdir.joinpath("results.json").read_text(encoding="utf-8"))
    rows = []
    for dataset_name, dataset in data.items():
        for scheduler, stats in dataset.items():
            for stat, value in stats.items():
                rows.append({
                    "dataset": dataset_name,
                    "scheduler": scheduler,
                    "stat": stat,
                    "value": value,
                })

    df_stats = pd.DataFrame(rows).set_index(["dataset", "scheduler", "stat"])
    print(df_stats.unstack())

    # Box plot: x-axis is scheduler, y-axis is makespan, color is dataset
    fig = px.box(
        df_stats.reset_index(),
        x="scheduler", y="value", color="dataset",
        template="plotly_white",
        title="Makespan of Schedulers on Datasets"
    )
    # make higher resolution
    fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
    )
    fig.write_image(thisdir.joinpath("boxplot.png"))

if __name__ == "__main__":
    main()
