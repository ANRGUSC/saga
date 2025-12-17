# Importing required libraries to load and examine the data
import logging
import pathlib
import pandas as pd

from saga.utils.draw import gradient_heatmap

from common import resultsdir, outputdir

DATASET_ORDER = [
    "in_trees", "out_trees", "chains",

    "blast", "bwa", "cycles", "epigenomics",
    "genome", "montage", "seismology", "soykb",
    "srasearch",

    "etl", "predict", "stats", "train",
]

SCHEDULER_RENAMES = {
    "Cpop": "CPoP",
    "Heft": "HEFT",
}

def load_data(resultsdir: pathlib.Path, glob: str | None = None) -> pd.DataFrame:
    data = None
    glob = glob or "*.csv"
    for path in resultsdir.glob(glob):
        df_dataset = pd.read_csv(path, index_col=0)
        df_dataset["dataset"] = path.stem
        if data is None:
            data = df_dataset
        else:
            data = pd.concat([data, df_dataset], ignore_index=True)
    if data is None:
        return pd.DataFrame()
    return data

def run_analysis(resultsdir: pathlib.Path,
                 outputdir: pathlib.Path,
                 glob: str | None= None,
                 title: str | None = None,
                 upper_threshold: float = 5.0) -> None:
    """Analyze the results."""
    outputdir.mkdir(parents=True, exist_ok=True)
    data = load_data(resultsdir, glob)
    if data.empty:
        logging.info("No data found. Skipping.")
        return

    data["Scheduler"] = data["Scheduler"].str.replace("Scheduler", "")
    data["Scheduler"] = data["Scheduler"].replace(SCHEDULER_RENAMES)

    data["Makespan Ratio"] = data.groupby(by=["dataset", "Instance"])["Makespan"].transform(
        lambda x: x / x.min()
    )

    print(data)

    ax = gradient_heatmap(
        data,
        x="Scheduler",
        y="dataset",
        color="Makespan Ratio",
        cmap="coolwarm",
        upper_threshold=upper_threshold,
        title=title,
        x_label="Scheduler",
        y_label="Dataset",
        color_label="Maximum Makespan Ratio"
    )
    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(
            outputdir.joinpath("benchmarking.pdf"),
            dpi=300,
            bbox_inches='tight'
        )

    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(
            outputdir.joinpath("benchmarking.png"),
            dpi=300,
            bbox_inches='tight'
        )


def main():
    run_analysis(
        resultsdir=resultsdir,
        outputdir=outputdir,
        title="Benchmarking of SAGA Schedulers",
        upper_threshold=5.0
    )

if __name__ == "__main__":
    main()