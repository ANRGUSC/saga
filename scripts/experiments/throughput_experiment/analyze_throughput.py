import logging
import pathlib

from saga.utils.draw import gradient_heatmap
from scripts.experiments.benchmarking.analyze import load_data


thisdir = pathlib.Path(__file__).parent.resolve()
resultsdir = thisdir / "results" / "throughput"
outputdir = thisdir / "output" / "throughput"

SCHEDULER_RENAMES = {
    "Cpop": "CPoP",
    "Heft": "HEFT",
}


def run_analysis(
    resultsdir: pathlib.Path,
    outputdir: pathlib.Path,
    glob: str | None = None,
    title: str | None = None,
    upper_threshold: float = 5.0,
) -> None:
    outputdir.mkdir(parents=True, exist_ok=True)
    data = load_data(resultsdir, glob)
    if data.empty:
        logging.info("No data found. Skipping.")
        return

    data["Scheduler"] = data["Scheduler"].str.replace("Scheduler", "")
    data["Scheduler"] = data["Scheduler"].replace(SCHEDULER_RENAMES)

    # Makespan ratio: 1.0 = best (lowest makespan)
    data["Makespan Ratio"] = data.groupby(["dataset", "Instance"])["Makespan"].transform(
        lambda x: x / x.min()
    )

    # Throughput ratio: 1.0 = best (highest throughput)
    data["Throughput Ratio"] = data.groupby(["dataset", "Instance"])["Throughput"].transform(
        lambda x: x.max() / x
    )

    makespan_title = f"{title} — Makespan" if title else "Makespan"
    ax = gradient_heatmap(
        data,
        x="Scheduler",
        y="dataset",
        color="Makespan Ratio",
        cmap="coolwarm",
        upper_threshold=upper_threshold,
        title=makespan_title,
        x_label="Scheduler",
        y_label="Dataset",
        color_label="Maximum Makespan Ratio",
    )
    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(outputdir / "makespan_benchmarking.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(outputdir / "makespan_benchmarking.png", dpi=300, bbox_inches="tight")

    throughput_title = f"{title} — Throughput" if title else "Throughput"
    ax = gradient_heatmap(
        data,
        x="Scheduler",
        y="dataset",
        color="Throughput Ratio",
        cmap="coolwarm",
        upper_threshold=upper_threshold,
        title=throughput_title,
        x_label="Scheduler",
        y_label="Dataset",
        color_label="Maximum Throughput Ratio",
    )
    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(outputdir / "throughput_benchmarking.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(outputdir / "throughput_benchmarking.png", dpi=300, bbox_inches="tight")


def main():
    run_analysis(
        resultsdir=resultsdir,
        outputdir=outputdir,
        title="Throughput Benchmarking",
        upper_threshold=5.0,
    )


if __name__ == "__main__":
    main()
