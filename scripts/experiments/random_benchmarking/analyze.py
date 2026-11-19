"""Analyze the priority-function experiment.

Each randomized scheduler is compared against the deterministic baseline that shares its
insertion policy. For every instance the baseline is a single makespan and the randomized
scheduler is a distribution of makespans (one per sample), so the question "how much does
the priority function matter" becomes "where does the baseline sit in that distribution".

Outputs (in ``outputdir``):
    summary.csv / summary.md   per-dataset statistics of the ratio random / baseline
    per_instance.csv           the same statistics for every instance
    ratio_distributions.{pdf,png}  boxplots of the ratio per dataset, log scale, with
                                   whiskers at the minimum and maximum, and a caption
"""

import logging
import pathlib
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from common import outputdir, resultsdir

DATASET_ORDER = [
    "in_trees", "out_trees", "chains",
    "blast", "bwa", "cycles", "epigenomics",
    "genome", "montage", "seismology", "soykb",
    "srasearch",
    "etl", "predict", "stats", "train",
]

# randomized scheduler -> deterministic baseline with the same insertion policy
BASELINES: Dict[str, str] = {
    "HEFT_RandomOrder": "HEFT",
    "CPoP_RandomOrder": "CPoP",
    "RandomOrder_RandomNode": "HEFT",
}


def load_data(resultsdir: pathlib.Path, glob: Optional[str] = None) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(resultsdir.glob(glob or "*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def dataset_key(name: str) -> int:
    return DATASET_ORDER.index(name) if name in DATASET_ORDER else len(DATASET_ORDER)


def ratios_to_baseline(data: pd.DataFrame) -> pd.DataFrame:
    """Rows of the randomized schedulers with their makespan divided by the baseline's
    makespan on the same instance."""
    baseline_makespan = (
        data[data["Scheduler"].isin(set(BASELINES.values()))]
        .set_index(["Dataset", "Instance", "Scheduler"])["Makespan"]
    )
    frames = []
    for scheduler, baseline in BASELINES.items():
        rows = data[data["Scheduler"] == scheduler].copy()
        if rows.empty:
            continue
        idx = pd.MultiIndex.from_arrays(
            [rows["Dataset"], rows["Instance"], [baseline] * len(rows)]
        )
        rows["Baseline"] = baseline
        rows["Baseline Makespan"] = baseline_makespan.reindex(idx).to_numpy()
        rows["Ratio"] = rows["Makespan"] / rows["Baseline Makespan"]
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def per_instance_summary(ratios: pd.DataFrame) -> pd.DataFrame:
    grouped = ratios.groupby(["Dataset", "Scheduler", "Baseline", "Instance"])
    summary = grouped["Ratio"].agg(
        samples="size",
        median_ratio="median",
        best_ratio="min",
        worst_ratio="max",
        frac_random_better=lambda r: (r < 1 - 1e-9).mean(),
    ).reset_index()
    return summary


def per_dataset_summary(instances: pd.DataFrame) -> pd.DataFrame:
    grouped = instances.groupby(["Dataset", "Scheduler", "Baseline"])
    summary = grouped.agg(
        instances=("Instance", "size"),
        median_ratio=("median_ratio", "mean"),
        best_ratio=("best_ratio", "mean"),
        worst_ratio=("worst_ratio", "mean"),
        frac_random_better=("frac_random_better", "mean"),
        frac_instances_any_better=("best_ratio", lambda b: (b < 1 - 1e-9).mean()),
    ).reset_index()
    summary["_order"] = summary["Dataset"].map(dataset_key)
    return summary.sort_values(["Scheduler", "_order"]).drop(columns="_order")


DISTRIBUTION_CAPTION = (
    "Each sample's makespan is divided by the makespan of the baseline that shares its "
    "insertion policy on the same instance, and each box pools every sample from every "
    "instance in the dataset. Boxes span the 25th to 75th percentile, the blue line marks "
    "the median, and the whiskers reach the minimum and maximum. The dashed line marks a "
    "ratio of 1: below it the randomized scheduler beat the baseline, above it the baseline "
    "won. The vertical axis is logarithmic, so a ratio of 0.5 and a ratio of 2 are equally "
    "far from 1."
)


def plot_distributions(ratios: pd.DataFrame, outputdir: pathlib.Path):
    """Boxplots of the makespan ratio (random / baseline) per dataset, one panel per
    randomized scheduler, on a log scale so 0.5 and 2 are equally far from 1."""
    schedulers = [s for s in BASELINES if s in set(ratios["Scheduler"])]
    datasets = sorted(ratios["Dataset"].unique(), key=dataset_key)
    fig, axes = plt.subplots(1, len(schedulers), figsize=(5 * len(schedulers), 4.5), squeeze=False)
    for col, scheduler in enumerate(schedulers):
        baseline = BASELINES[scheduler]
        ax = axes[0, col]
        samples = [
            ratios.loc[(ratios["Scheduler"] == scheduler) & (ratios["Dataset"] == d), "Ratio"]
            for d in datasets
        ]
        # Box: 25th to 75th percentile; line: median; whiskers: minimum and maximum.
        ax.boxplot(samples, tick_labels=datasets, whis=(0, 100), showfliers=False,
                   widths=0.6, medianprops={"color": "#1f4e79", "linewidth": 2})
        ax.axhline(1.0, color="#777777", linewidth=1, linestyle="--")
        ax.set_yscale("log")
        ax.set_title(f"{scheduler} / {baseline}")
        ax.set_ylabel("makespan ratio (random / baseline), log" if col == 0 else "")
        ax.tick_params(axis="x", rotation=45)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.text(0.01, -0.02, DISTRIBUTION_CAPTION, ha="left", va="top", wrap=True,
             fontsize=10, color="#52514e")
    for ext in ("pdf", "png"):
        fig.savefig(outputdir / f"ratio_distributions.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (floats to three decimals)."""
    cells = [
        [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        for row in df.itertuples(index=False)
    ]
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(" --- " for _ in df.columns) + "|"
    return "\n".join([header, sep] + ["| " + " | ".join(r) + " |" for r in cells])


def run_analysis(
    resultsdir: pathlib.Path,
    outputdir: pathlib.Path,
    glob: Optional[str] = None,
) -> None:
    outputdir.mkdir(parents=True, exist_ok=True)
    data = load_data(resultsdir, glob)
    if data.empty:
        logging.info("No data found. Skipping.")
        return

    ratios = ratios_to_baseline(data)
    missing = ratios["Baseline Makespan"].isna().sum()
    if missing:
        logging.warning("%d randomized rows have no baseline row; dropping them.", missing)
        ratios = ratios.dropna(subset=["Baseline Makespan"])

    instances = per_instance_summary(ratios)
    summary = per_dataset_summary(instances)

    instances.to_csv(outputdir / "per_instance.csv", index=False)
    summary.to_csv(outputdir / "summary.csv", index=False)
    markdown = to_markdown(summary)
    (outputdir / "summary.md").write_text(markdown + "\n")
    print(markdown)

    plot_distributions(ratios, outputdir)


def main():
    run_analysis(resultsdir=resultsdir, outputdir=outputdir)


if __name__ == "__main__":
    main()
