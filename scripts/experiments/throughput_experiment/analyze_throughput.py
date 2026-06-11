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
        avg_throughput = group.groupby("Scheduler")["Throughput"].mean()
        best = avg_throughput.max()
        diff = (avg_throughput - best).sort_values()

        fig, ax = plt.subplots(figsize=(6, 3))
        diff.plot(kind="bar", ax=ax, color="steelblue", width=0.7)
        ax.axhline(0, color="black", linewidth=0.6)

        dataset_title = f"{title} — {dataset}" if title else str(dataset)
        ax.set_title(dataset_title, fontsize=7)
        ax.set_xlabel("Scheduler", fontsize=6)
        ax.set_ylabel("Throughput delta vs best (jobs/s)", fontsize=6)
        ax.tick_params(axis="both", labelsize=5)
        plt.xticks(rotation=45, ha="right")
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
