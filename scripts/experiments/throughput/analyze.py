"""Analyze throughput results: per-workflow gradient heatmaps.

Reads results/<branch>_<regime>.csv (from run.py), normalizes throughput and makespan
per instance against the best config on that instance, then writes:
  - output/<branch>_<regime>/<workflow>_throughput.pdf : mean throughput ratio, schedulers x CCR.
  - output/<branch>_<regime>/<workflow>_makespan.pdf   : mean makespan ratio, schedulers x CCR.

In both cases 1.0 is the best config on that instance, but the direction flips: throughput
is better when higher, so ThroughputRatio = Throughput / max(Throughput) <= 1 (lower is
worse); makespan is better when lower, so MakespanRatio = Makespan / min(Makespan) >= 1
(higher is worse).

Usage:
    python analyze.py riotbench deterministic
    python analyze.py wfcommons stochastic
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

from common import resultsdir, outputdir
from saga.utils.draw import gradient_heatmap

# Figures are emitted as PDF: vector output scales to the column width without
# resampling, and the labels stay real text in the typeset paper. fonttype 42
# embeds TrueType rather than Type 3, which some publishers reject.
FIG_EXT = "pdf"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# per-realization identity: the best config is chosen within one of these groups
INSTANCE_KEYS = ["Workflow", "CCR", "Instance", "Seed"]

_POLICY_RANK = {
    "reschedule": 0,
    "conditional": 1,
    "random50": 2,
    "random25": 3,
    "random10": 4,
    "random5": 5,
    "random1": 6,
    "checkpoint_10": 7,
    "checkpoint_quarterly": 8,
    "checkpoint_mid": 9,
    "static": 10,
}
_STANDALONE_RANK = {"FastestNode": 0, "MaxTP": 1}

# Display names, matching the paper: the throughput variants are T-<base>, the
# standalone throughput heuristic is MaxT, and policies are title-cased.
_POLICY_DISPLAY = {
    "reschedule": "Reschedule", "conditional": "Conditional",
    "random50": "Random-50%", "random25": "Random-25%", "random10": "Random-10%",
    "random5": "Random-5%", "random1": "Random-1%",
    "checkpoint_10": "Checkpoint-10%", "checkpoint_quarterly": "Checkpoint-Qtr",
    "checkpoint_mid": "Checkpoint-Mid", "static": "Static",
}


def scheduler_display(name: str) -> str:
    """Raw scheduler name -> paper-facing label (see _POLICY_DISPLAY)."""
    if name == "MaxTP":
        return "MaxT"
    if name in _STANDALONE_RANK:
        return name
    base, policy = name.split("_", 1)
    if base == "MaxTP":
        base = "MaxT"
    elif base.endswith("-Tp"):
        base = f"T-{base[:-3]}"
    return f"{base}-{_POLICY_DISPLAY.get(policy, policy)}"


def scheduler_order(name: str):
    """Sort key placing throughput bases above EFT, HEFT above CPoP, reschedule above
    conditional above the random policies above static, with FastestNode/MaxT last.
    Smaller sorts toward the top row. Takes a display name (see scheduler_display)."""
    if name in ("FastestNode", "MaxT"):
        return (3, 0, 0 if name == "FastestNode" else 1)
    base, _, policy = name.partition("-")
    if base == "T":  # "T-HEFT-Reschedule" -> base "T-HEFT"
        algo_name, _, policy = policy.partition("-")
        comparator, algo = 0, (0 if algo_name == "HEFT" else 1)
    elif base == "MaxT":
        comparator, algo = 1, 0
    else:
        comparator, algo = 2, (0 if base == "HEFT" else 1)
    return (comparator, algo, _POLICY_DISPLAY_RANK.get(policy, 99))


_POLICY_DISPLAY_RANK = {_POLICY_DISPLAY[k]: v for k, v in _POLICY_RANK.items()}


def load(branch: str, regime: str) -> pd.DataFrame:
    path = resultsdir / f"{branch}_{regime}.csv"
    if not path.exists():
        available = sorted(p.stem for p in resultsdir.glob("*.csv"))
        raise SystemExit(
            f"No results at {path.name}. Available: {available or '(none)'}. "
            f"Usage: python analyze.py <branch> <regime>  (regime = deterministic|stochastic)"
        )
    df = pd.read_csv(path)
    best_throughput = df.groupby(INSTANCE_KEYS)["Throughput"].transform("max")
    df["ThroughputRatio"] = df["Throughput"] / best_throughput
    best_makespan = df.groupby(INSTANCE_KEYS)["Makespan"].transform("min")
    df["MakespanRatio"] = df["Makespan"] / best_makespan
    return df


# Policies shown in the heatmaps. The full grid is 11 policies x 5 bases plus
# FastestNode, which is 56 rows and illegible at column width; the policy breakdown
# is reported separately by figure_scripts.py. Static and Reschedule bracket the
# range, which is what the per-workflow claims in the paper rest on.
_HEATMAP_POLICIES = ("static", "reschedule")


def heatmaps(df: pd.DataFrame, branch: str, regime: str) -> None:
    outdir = outputdir / f"{branch}_{regime}"
    outdir.mkdir(parents=True, exist_ok=True)
    keep = df["Scheduler"].isin(_STANDALONE_RANK) | df["Scheduler"].str.endswith(
        tuple(f"_{p}" for p in _HEATMAP_POLICIES)
    )
    df = df[keep]
    for workflow, group in df.groupby("Workflow"):
        # Pass the raw per-instance/seed rows so each cell renders a gradient over its
        # distribution (aggregating to one value per cell would flatten it); the cell
        # label is the mean.
        for metric, ratio_col, cmap in (
            ("throughput", "ThroughputRatio", "coolwarm_r"),  # high (good) is cool, low is warm/red
            ("makespan", "MakespanRatio", "coolwarm"),  # low (good) is cool, high is warm/red
        ):
            cell = group[["Scheduler", "CCR", ratio_col]].copy()
            cell["Scheduler"] = cell["Scheduler"].map(scheduler_display)
            ax = gradient_heatmap(
                cell, x="CCR", y="Scheduler", color=ratio_col,
                title=f"{branch} / {regime}: {workflow} ({metric})",
                x_label="CCR", y_label="scheduler", color_label=f"{metric} ratio (mean)",
                yorder=scheduler_order,
                cmap=cmap,
                cell_font_size=14, font_size=14,
                figsize=(9, max(4.0, 0.55 * cell["Scheduler"].nunique() + 1.5)),
            )
            ax.get_figure().savefig(outdir / f"{workflow}_{metric}.{FIG_EXT}", bbox_inches="tight", dpi=120)
            plt.close(ax.get_figure())
    print(f"heatmaps -> {outdir}")


def main() -> None:
    branch = sys.argv[1] if len(sys.argv) > 1 else "riotbench"
    regime = sys.argv[2] if len(sys.argv) > 2 else "deterministic"
    df = load(branch, regime)
    heatmaps(df, branch, regime)


if __name__ == "__main__":
    main()
