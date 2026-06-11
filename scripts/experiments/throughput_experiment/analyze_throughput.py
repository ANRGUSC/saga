import logging
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


thisdir = pathlib.Path(__file__).parent.resolve()
resultsdir = thisdir / "results" / "throughput"
outputdir = thisdir / "output" / "throughput"

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


def run_analysis(
    resultsdir: pathlib.Path,
    outputdir: pathlib.Path,
    glob: str | None = None,
    title: str | None = None,
) -> None:
    outputdir.mkdir(parents=True, exist_ok=True)
    data = load_data(resultsdir, glob)
    if data.empty:
        logging.info("No data found. Skipping.")
        return

    data["Scheduler"] = data["Scheduler"].str.replace("Scheduler", "")
    data["Scheduler"] = data["Scheduler"].replace(SCHEDULER_RENAMES)

    for dataset, group in data.groupby("dataset"):
        best_mean = group.groupby("Scheduler")["Throughput"].mean().max()
        group = group.copy()
        group["ratio"] = group["Throughput"] / best_mean

        order = (
            group.groupby("Scheduler")["ratio"]
            .median()
            .sort_values()
            .index.tolist()
        )
        plot_data = [group.loc[group["Scheduler"] == s, "ratio"].values for s in order]

        fig, ax = plt.subplots(figsize=(6, max(3, len(order) * 0.3)))
        ax.boxplot(
            plot_data,
            vert=False,
            patch_artist=True,
            labels=order,
            medianprops={"color": "black", "linewidth": 1.2},
            boxprops={"facecolor": "steelblue", "alpha": 0.7},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.5},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )

        dataset_title = f"{title} — {dataset}" if title else str(dataset)
        ax.set_title(dataset_title, fontsize=7)
        ax.set_xlabel("Throughput ratio vs best (1.0 = best)", fontsize=6)
        ax.tick_params(axis="both", labelsize=6)
        fig.tight_layout(pad=0.5)

        safe_name = str(dataset).replace("/", "_")
        fig.savefig(outputdir / f"{safe_name}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(outputdir / f"{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    run_analysis(
        resultsdir=resultsdir,
        outputdir=outputdir,
        title="Throughput Benchmarking",
    )


if __name__ == "__main__":
    main()
