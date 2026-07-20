"""Analyze throughput results: per-workflow gradient heatmaps.

Reads results/<branch>_<regime>.csv (from run.py), normalizes throughput per instance
against the best config on that instance, then writes:
  - output/<branch>_<regime>/<workflow>.png : median throughput ratio, schedulers x CCR.

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

_POLICY_RANK = {
    "reschedule": 0,
    "conditional": 1,
    "random50": 2,
    "random25": 3,
    "random10": 4,
    "static": 5,
}
_STANDALONE_RANK = {"FastestNode": 0, "MaxTP": 1}


def scheduler_order(name: str):
    """Sort key placing throughput bases above EFT, HEFT above CPoP, reschedule above
    conditional above the random policies above static, with FastestNode/MaxTP last.
    Smaller sorts toward the top row."""
    if name in _STANDALONE_RANK:
        return (2, 0, _STANDALONE_RANK[name])
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


def main() -> None:
    branch = sys.argv[1] if len(sys.argv) > 1 else "riotbench"
    regime = sys.argv[2] if len(sys.argv) > 2 else "deterministic"
    df = load(branch, regime)
    heatmaps(df, branch, regime)


if __name__ == "__main__":
    main()
