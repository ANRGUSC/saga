# Importing required libraries to load and examine the data
import logging
import pathlib

import pandas as pd
from saga.experiment.plot import gradient_heatmap


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

def load_data(resultsdir, glob: str = None) -> pd.DataFrame:
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

def run(resultsdir: pathlib.Path,
        outputdir: pathlib.Path,
        glob: str = None,
        title: str = None,
        upper_threshold: float = 5.0) -> None:
    """Analyze the results."""
    outputdir.mkdir(parents=True, exist_ok=True)
    data = load_data(resultsdir, glob)
    if data.empty:
        logging.info("No data found. Skipping.")
        return
    data["scheduler"] = data["scheduler"].str.replace("Scheduler", "")
    data["scheduler"] = data["scheduler"].replace(SCHEDULER_RENAMES)

    ax = gradient_heatmap(
        data,
        x="scheduler",
        y="dataset",
        color="makespan_ratio",
        cmap="coolwarm",
        upper_threshold=upper_threshold,
        title=title,
        x_label="Scheduler",
        y_label="Dataset",
        color_label="Maximum Makespan Ratio"
    )
    ax.get_figure().savefig(
        outputdir.joinpath("benchmarking.pdf"),
        dpi=300,
        bbox_inches='tight'
    )
    ax.get_figure().savefig(
        outputdir.joinpath("benchmarking.png"),
        dpi=300,
        bbox_inches='tight'
    )
