# Importing required libraries to load and examine the data
import logging
import pathlib
import re

import pandas as pd
from saga.experiment.plot import gradient_heatmap
from saga.experiment import resultsdir, outputdir

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

def run(resultspath: pathlib.Path,
        savepath: pathlib.Path,
        title: str = None,
        upper_threshold: float = 5.0,
        sample_agg_mode: str = "mean") -> None:
    """Analyze the results."""
    df = pd.read_csv(resultspath, index_col=0)
    df = df.groupby(["dataset", "instance", "scheduler"]).agg({"makespan": sample_agg_mode}).reset_index()

    best_makespan = df.groupby(["dataset", "instance"]).agg({"makespan": "min"}).rename(columns={"makespan": "best_makespan"})
    df = df.join(best_makespan, on=["dataset", "instance"])
    df["makespan_ratio"] = df["makespan"] / df["best_makespan"]

    df["scheduler"] = df["scheduler"].str.replace("Scheduler", "")
    for old, new in SCHEDULER_RENAMES.items():
        df["scheduler"] = df["scheduler"].str.replace(old, new)


    re_dataset = re.compile(r"^(?P<dataset_name>.+)_ccr_(?P<ccr_mean>\d(?:\.\d+)?)_std_(?P<ccr_std>\d(?:\.\d+)?)$")
    df_dataset = df["dataset"].str.extract(re_dataset)
    df_dataset["ccr_mean"] = df_dataset["ccr_mean"].astype(float).round(2)
    df_dataset["ccr_std"] = df_dataset["ccr_std"].astype(float).round(2)
    df = df.join(df_dataset).drop(columns="dataset")
    df = df[['dataset_name', 'ccr_mean', 'ccr_std', 'instance', 'scheduler', 'makespan_ratio']]

    ax = gradient_heatmap(
        df,
        y="scheduler",
        x=["dataset_name", "ccr_mean", "ccr_std"],
        color="makespan_ratio",
        cmap="coolwarm",
        upper_threshold=upper_threshold,
        title=title,
        y_label="Scheduler[determinizer]",
        x_label="Dataset/Comm Mean/Comm STD",
        color_label="Maximum Makespan Ratio",
        figsize=(14,8),
    )

    savepath.parent.mkdir(parents=True, exist_ok=True)
    ax.get_figure().savefig(savepath.with_suffix(".png"), dpi=300, bbox_inches='tight')
    ax.get_figure().savefig(savepath.with_suffix(".pdf"), dpi=300, bbox_inches='tight')

def main():
    outputdir_stochastic = outputdir / "stochastic_benchmarking"
    resultspath_stochastic = resultsdir / "stochastic_benchmarking/results.csv"
    run(
        resultspath_stochastic,
        outputdir_stochastic.joinpath("stochastic_benchmarking_mean.png"),
        title="Stochastic Benchmarking (Average Performance)",
        upper_threshold=5.0,
        sample_agg_mode="mean"
    )
    run(
        resultspath_stochastic,
        outputdir_stochastic.joinpath("stochastic_benchmarking_max.png"),
        title="Stochastic Benchmarking (Worst-Case Performance)",
        upper_threshold=5.0,
        sample_agg_mode="max"
    )

if __name__ == "__main__":
    main()