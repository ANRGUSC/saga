# Importing required libraries to load and examine the data
import pathlib

import pandas as pd
import plotly.express as px

thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    """Analyze the results."""
    data = pd.read_csv(thisdir.joinpath("results.csv"), index_col=0)

    # strip "Scheduler" from scheduler names
    data["scheduler"] = data["scheduler"].str.replace("Scheduler", "")

    # Box Plot of Makespan, x-axis: scheduler, color: dataset
    fig = px.box(
        data, x="scheduler", y="makespan_ratio", color="dataset",
        template="plotly_white",
        labels={
            "scheduler": "Scheduler",
            "makespan_ratio": "Makespan Ratio",
            "dataset": "Dataset",
        }
    )
    max_val = data["makespan_ratio"].max()
    # # get number of digits in max_val
    num_digits = len(str(int(max_val)))
    # # round to next 10^num_digits
    max_val = int(max_val / 10**(num_digits - 1) + 1) * 10**(num_digits - 1)
    # tick_jump = 10**(num_digits - 1)
    # ticks = [1] + [tick_jump * i for i in range(1, int(max_val / tick_jump) + 1)]

    fig.update_yaxes(range=[1/2, max_val])
    # set first tick to 1 then 10, 20, 30, ...
    # fig.update_yaxes(tickvals=ticks)
    fig.write_html(thisdir.joinpath("boxplot.html"))
    fig.write_image(thisdir.joinpath("boxplot.png"))




if __name__ == "__main__":
    main()
