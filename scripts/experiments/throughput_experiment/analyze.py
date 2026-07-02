"""Analyze throughput results: per-workflow gradient heatmaps and the TP policy ladder.

Reads results/<branch>_<regime>.csv (from run.py), normalizes throughput per instance
against the best config on that instance, then writes:
  - output/<branch>_<regime>/<workflow>.png : median throughput ratio, schedulers x CCR.
  - a printed TP ladder (static / inspirit / reschedule within the throughput bases),
    median ratio and win-rate against static.

Usage:
    python analyze.py riotbench deterministic
    python analyze.py wfcommons stochastic
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

from common import resultsdir, outputdir
from saga.utils.draw import gradient_heatmap

# per-realization identity: the best config is chosen within one of these groups
INSTANCE_KEYS = ["Workflow", "CCR", "Instance", "Seed"]

_POLICY_RANK = {"reschedule": 0, "inspirit": 1, "static": 2}


def scheduler_order(name: str):
    """Sort key placing throughput bases above EFT, HEFT above CPoP, reschedule above
    inspirit above static, with FastestNode last. Smaller sorts toward the top row."""
    if name == "FastestNode":
        return (2, 0, 0)
    base, policy = name.split("_", 1)
    comparator = 0 if base.endswith("-Tp") else 1
    algo = 0 if base.startswith("HEFT") else 1
    return (comparator, algo, _POLICY_RANK.get(policy, 3))


def load(branch: str, regime: str) -> pd.DataFrame:
    path = resultsdir / f"{branch}_{regime}.csv"
    if not path.exists():
        available = sorted(p.stem for p in resultsdir.glob("*.csv"))
        raise SystemExit(
            f"No results at {path.name}. Available: {available or '(none)'}. "
            f"Usage: python analyze.py <branch> <regime>  (regime = deterministic|stochastic)"
        )
    df = pd.read_csv(path)
    best = df.groupby(INSTANCE_KEYS)["Throughput"].transform("max")
    df["Ratio"] = df["Throughput"] / best
    return df


def heatmaps(df: pd.DataFrame, branch: str, regime: str) -> None:
    outdir = outputdir / f"{branch}_{regime}"
    outdir.mkdir(parents=True, exist_ok=True)
    for workflow, group in df.groupby("Workflow"):
        # Pass the raw per-instance/seed rows so each cell renders a gradient over its
        # distribution (aggregating to one value per cell would flatten it); the cell
        # label is the mean.
        cell = group[["Scheduler", "CCR", "Ratio"]]
        ax = gradient_heatmap(
            cell, x="CCR", y="Scheduler", color="Ratio",
            title=f"{branch} / {regime}: {workflow}",
            x_label="CCR", y_label="scheduler", color_label="throughput ratio (median)",
            yorder=scheduler_order,
            cmap="coolwarm_r",  # reverse so high throughput (good) is cool, low is warm/red
            cell_font_size=14, font_size=14, figsize=(9, 7),
        )
        ax.get_figure().savefig(outdir / f"{workflow}.png", bbox_inches="tight", dpi=120)
        plt.close(ax.get_figure())
    print(f"heatmaps -> {outdir}")


def ladder(df: pd.DataFrame) -> None:
    """Median ratio and win-rate against static, within the throughput bases."""
    tp = df[df["Scheduler"].str.contains("-Tp_")].copy()
    if tp.empty:
        return
    tp[["Base", "Policy"]] = tp["Scheduler"].str.split("_", n=1, expand=True)
    static = (
        tp[tp["Policy"] == "static"]
        .set_index(INSTANCE_KEYS + ["Base"])["Throughput"]
        .rename("StaticThroughput")
    )
    merged = tp.join(static, on=INSTANCE_KEYS + ["Base"])
    merged["beats_static"] = merged["Throughput"] > merged["StaticThroughput"] + 1e-12

    summary = merged.groupby("Policy").agg(
        median_ratio=("Ratio", "median"),
        win_rate_vs_static=("beats_static", "mean"),
    )
    order = ["static", "inspirit", "reschedule"]
    summary = summary.reindex([p for p in order if p in summary.index])
    print("\nTP policy ladder (throughput bases: HEFT-Tp, CPoP-Tp)\n")
    print(summary.round(3).to_string())


def main() -> None:
    branch = sys.argv[1] if len(sys.argv) > 1 else "riotbench"
    regime = sys.argv[2] if len(sys.argv) > 2 else "deterministic"
    df = load(branch, regime)
    heatmaps(df, branch, regime)
    ladder(df)


if __name__ == "__main__":
    main()
