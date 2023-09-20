import pathlib
import pandas as pd
import plotly.express as px
from analyze import gradient_heatmap

thisdir = pathlib.Path(__file__).parent.absolute()
thisdir.joinpath("results_hybrid")

def load_results() -> pd.DataFrame:
    data = None
    for path in thisdir.glob("results_hybrid/*/*.csv"):
        dataset_name = path.parent.stem
        df_dataset = pd.read_csv(path, index_col=0)
        df_dataset["dataset"] = dataset_name

        # scheduler is of form <name>.<num> split this into two columns
        df_dataset["num_algs"] = df_dataset["scheduler"].str.split(".", expand=True)[1].astype(int) + 1
        df_dataset["scheduler"] = df_dataset["scheduler"].str.split(".", expand=True)[0]

        if data is None:
            data = df_dataset
        else:
            data = pd.concat([data, df_dataset], ignore_index=True)
    return data

def main():
    data = load_results()
    print(data)

    # remove all schedulers with MinMax in name
    data = data[~data["scheduler"].str.contains("MinMax")]

    ax = gradient_heatmap(
        data=data,
        x="dataset",
        y=["scheduler", "num_algs"],
        color="makespan_ratio"
    )
    ax.get_figure().savefig(
        thisdir.joinpath("hybrid_heatmap.png"),
        bbox_inches="tight",
        dpi=300,
    )

    # plot the results
    # box plot with x-axis "num_algs", y-axis "makespan_ratio", category colors "scheduler"
    fig = px.box(
        data,
        x="num_algs",
        y="makespan_ratio",
        color="scheduler",
        template="plotly_white",
        facet_col="dataset",
        facet_col_wrap=2,
    )
    # fig.update_yaxes(matches=None)
    fig.write_html(thisdir.joinpath("hybrid_boxplot.html"))

    data_mean = data.groupby(["scheduler", "num_algs", "dataset"]).mean().reset_index()
    fig = px.line(
        data_mean,
        x="num_algs",
        y="makespan_ratio",
        color="scheduler",
        template="plotly_white",
        facet_col="dataset",
        facet_col_wrap=2,
    )
    # fig.update_yaxes(matches=None)
    fig.write_html(thisdir.joinpath("hybrid_lineplot.html"))

if __name__ == "__main__":
    main()
