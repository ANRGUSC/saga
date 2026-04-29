"""Visualize conditional scheduling experiment results.

Reads results.csv and summary_by_ctg.csv from one or more run folders and
generates charts into a visualizations/ directory.

Charts are saved under:
    visualizations/<timestamp>/
        trace_level/      — per-trace breakdowns from results.csv
        ctg_level/        — per-CTG aggregates from summary_by_ctg.csv
        expected_value/   — expected-value decomposition charts
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

thisdir = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_runs(run_dirs: List[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate results.csv and summary_by_ctg.csv from runs."""
    results_frames = []
    summary_frames = []

    for run_dir in run_dirs:
        run_name = run_dir.name
        r_path = run_dir / "results.csv"
        s_path = run_dir / "summary_by_ctg.csv"

        if r_path.exists():
            df = pd.read_csv(r_path)
            df["run"] = run_name
            results_frames.append(df)
        if s_path.exists():
            df = pd.read_csv(s_path)
            df["run"] = run_name
            summary_frames.append(df)

    results = pd.concat(results_frames, ignore_index=True) if results_frames else pd.DataFrame()
    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    return results, summary


# ---------------------------------------------------------------------------
# Trace-level charts (from results.csv)
# ---------------------------------------------------------------------------

def plot_online_vs_offline_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: trace_ctg_makespan (online) vs trace_standalone_makespan (offline).

    Points above the diagonal mean the CTG approach is slower for that trace.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    heuristics = df["heuristic"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(heuristics), 1)))

    for heur, color in zip(heuristics, colors):
        sub = df[df["heuristic"] == heur]
        ax.scatter(
            sub["trace_standalone_makespan"],
            sub["trace_ctg_makespan"],
            label=heur, alpha=0.7, color=color, edgecolors="black", linewidths=0.3,
        )

    lims = [0, max(df["trace_ctg_makespan"].max(), df["trace_standalone_makespan"].max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.4, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Standalone (offline) makespan")
    ax.set_ylabel("CTG (online) makespan")
    ax.set_title("Online vs Offline Makespan per Trace")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "online_vs_offline_scatter.pdf")
    plt.close(fig)


def plot_ratio_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram of trace_ratio_standalone_over_ctg across all traces."""
    fig, ax = plt.subplots(figsize=(10, 5))
    heuristics = df["heuristic"].unique()

    for heur in heuristics:
        sub = df[df["heuristic"] == heur]
        ax.hist(
            sub["trace_ratio_standalone_over_ctg"],
            bins=20, alpha=0.5, label=heur, edgecolor="black",
        )

    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.6, label="ratio = 1.0")
    ax.set_xlabel("Standalone / CTG ratio")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Standalone-over-CTG Ratio per Trace")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ratio_distribution.pdf")
    plt.close(fig)


def plot_ratio_by_heuristic_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    """Box plot: trace ratio grouped by heuristic."""
    fig, ax = plt.subplots(figsize=(10, 5))
    heuristics = sorted(df["heuristic"].unique())
    data = [df[df["heuristic"] == h]["trace_ratio_standalone_over_ctg"].dropna() for h in heuristics]

    bp = ax.boxplot(data, tick_labels=heuristics, patch_artist=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(heuristics)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Standalone / CTG ratio")
    ax.set_title("Trace Ratio by Heuristic")
    fig.tight_layout()
    fig.savefig(out_dir / "ratio_by_heuristic_boxplot.pdf")
    plt.close(fig)


def plot_ratio_by_network_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    """Box plot: trace ratio grouped by network."""
    fig, ax = plt.subplots(figsize=(10, 5))
    networks = sorted(df["network"].unique())
    data = [df[df["network"] == n]["trace_ratio_standalone_over_ctg"].dropna() for n in networks]

    bp = ax.boxplot(data, tick_labels=networks, patch_artist=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(networks)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Standalone / CTG ratio")
    ax.set_title("Trace Ratio by Network")
    fig.tight_layout()
    fig.savefig(out_dir / "ratio_by_network_boxplot.pdf")
    plt.close(fig)


def plot_gap_by_ctg(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: trace_gap per trace, grouped by CTG and colored by heuristic."""
    fig, ax = plt.subplots(figsize=(14, 5))
    df_sorted = df.sort_values(["ctg_id", "trace"])
    labels = [f"ctg{row.ctg_id}:{row.trace}" for _, row in df_sorted.iterrows()]

    heuristics = sorted(df["heuristic"].unique())
    colors = dict(zip(heuristics, plt.cm.tab10(np.linspace(0, 1, len(heuristics)))))
    bar_colors = [colors[h] for h in df_sorted["heuristic"]]

    ax.bar(range(len(df_sorted)), df_sorted["trace_gap"], color=bar_colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.4)
    ax.set_ylabel("Gap (CTG - Standalone)")
    ax.set_title("Trace Gap per Trace")

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=colors[h], label=h) for h in heuristics]
    ax.legend(handles=legend_handles)
    fig.tight_layout()
    fig.savefig(out_dir / "gap_by_trace.pdf")
    plt.close(fig)


def plot_probability_vs_gap(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: trace probability vs trace gap — does probability correlate?"""
    fig, ax = plt.subplots(figsize=(8, 6))
    heuristics = df["heuristic"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(heuristics), 1)))

    for heur, color in zip(heuristics, colors):
        sub = df[df["heuristic"] == heur]
        ax.scatter(
            sub["trace_probability"], sub["trace_gap"],
            label=heur, alpha=0.7, color=color, edgecolors="black", linewidths=0.3,
        )

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.4)
    ax.set_xlabel("Trace probability")
    ax.set_ylabel("Gap (CTG - Standalone)")
    ax.set_title("Trace Probability vs Gap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "probability_vs_gap.pdf")
    plt.close(fig)


def generate_trace_charts(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("  No trace-level data to plot.")
        return

    plot_online_vs_offline_scatter(df, out_dir)
    plot_ratio_distribution(df, out_dir)
    plot_ratio_by_heuristic_boxplot(df, out_dir)
    plot_ratio_by_network_boxplot(df, out_dir)
    plot_gap_by_ctg(df, out_dir)
    plot_probability_vs_gap(df, out_dir)
    print(f"  Trace-level charts saved to {out_dir}")


# ---------------------------------------------------------------------------
# Expected-value breakdown charts (from results.csv, one plot per CTG)
# ---------------------------------------------------------------------------

def _compute_weighted_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add weighted_contribution column: (online / offline) * P(trace)."""
    df = df.copy()
    df["weighted_contribution"] = (
        df["trace_ctg_makespan"] / df["trace_standalone_makespan"].clip(lower=1e-9)
    ) * df["trace_probability"]
    return df


def plot_stacked_weighted_contributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Stacked bar per (heuristic, network, CTG) showing each trace's
    weighted contribution to the expected ratio. Total height = expected ratio.
    """
    df = _compute_weighted_ratio(df)
    groups = df.groupby(["heuristic", "network", "ctg_id"])

    fig, ax = plt.subplots(figsize=(max(10, len(groups) * 1.5), 6))
    xlabels = []
    x_pos = 0
    cmap = plt.cm.Set2

    for (heur, net, cid), grp in sorted(groups):
        grp = grp.sort_values("weighted_contribution", ascending=False)
        bottom = 0.0
        for idx, (_, row) in enumerate(grp.iterrows()):
            color = cmap(idx / max(len(grp) - 1, 1))
            ax.bar(
                x_pos, row["weighted_contribution"], bottom=bottom,
                color=color, edgecolor="black", linewidth=0.3, width=0.7,
            )
            if row["weighted_contribution"] > 0.02:
                label_y = bottom + row["weighted_contribution"] / 2
                ax.text(
                    x_pos, label_y,
                    f"{row['trace']}\n({row['trace_probability']:.0%})",
                    ha="center", va="center", fontsize=6,
                )
            bottom += row["weighted_contribution"]

        xlabels.append(f"{heur}\n{net}\nctg{cid}")
        x_pos += 1

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="ratio = 1.0")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel("Cumulative weighted ratio")
    ax.set_title("Expected Ratio Breakdown: Weighted Contribution per Trace")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "stacked_weighted_contributions.pdf")
    plt.close(fig)


def plot_bubble_probability_ratio(df: pd.DataFrame, out_dir: Path) -> None:
    """Bubble chart: x = trace probability, y = online/offline ratio,
    bubble size = weighted contribution. Big bubbles top-right = impactful penalties.
    """
    df = _compute_weighted_ratio(df)
    df["ratio"] = df["trace_ctg_makespan"] / df["trace_standalone_makespan"].clip(lower=1e-9)

    fig, ax = plt.subplots(figsize=(10, 7))
    heuristics = df["heuristic"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(heuristics), 1)))

    for heur, color in zip(heuristics, colors):
        sub = df[df["heuristic"] == heur]
        sizes = sub["weighted_contribution"].abs() * 800 + 20
        ax.scatter(
            sub["trace_probability"], sub["ratio"],
            s=sizes, label=heur, alpha=0.6, color=color,
            edgecolors="black", linewidths=0.4,
        )

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Trace probability")
    ax.set_ylabel("Online / Offline ratio")
    ax.set_title("Bubble Chart: Probability × Ratio (size = weighted contribution)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bubble_probability_ratio.pdf")
    plt.close(fig)


def plot_expected_vs_mean_ratio(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    """Paired dot plot: expected_ratio vs mean_ratio per CTG.
    Lines connecting them show how probability weighting shifts the conclusion.
    """
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    heuristics = sorted(summary["heuristic"].unique())
    colors = dict(zip(heuristics, plt.cm.tab10(np.linspace(0, 1, len(heuristics)))))

    x_pos = 0
    xlabels = []
    xticks = []

    for _, row in summary.sort_values(["heuristic", "network", "ctg_id"]).iterrows():
        c = colors[row["heuristic"]]
        exp = row["expected_ratio"]
        mean = row["mean_ratio"]

        ax.plot([x_pos, x_pos], [exp, mean], color="grey", linewidth=1, zorder=1)
        ax.scatter(x_pos, exp, color=c, marker="o", s=80, zorder=2, edgecolors="black", linewidths=0.5)
        ax.scatter(x_pos, mean, color=c, marker="s", s=80, zorder=2, edgecolors="black", linewidths=0.5)

        xlabels.append(f"{row['heuristic']}\n{row['network']}\nctg{row['ctg_id']}")
        xticks.append(x_pos)
        x_pos += 1

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.4)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel("Ratio (online / offline)")
    ax.set_title("Expected Ratio (circle) vs Mean Ratio (square) per CTG")

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="grey", markerfacecolor="grey", label="Expected (prob-weighted)"),
        Line2D([0], [0], marker="s", color="grey", markerfacecolor="grey", label="Mean (unweighted)"),
    ]
    ax.legend(handles=legend_items)
    fig.tight_layout()
    fig.savefig(out_dir / "expected_vs_mean_ratio.pdf")
    plt.close(fig)


def plot_cumulative_expected_value(df: pd.DataFrame, out_dir: Path) -> None:
    """CDF-style plot: for each (heuristic, network, CTG), sort traces from
    worst to best ratio, plot cumulative sum of weighted_ratio. Shows how
    quickly the expected value accumulates.
    """
    df = _compute_weighted_ratio(df)
    df["ratio"] = df["trace_ctg_makespan"] / df["trace_standalone_makespan"].clip(lower=1e-9)
    groups = df.groupby(["heuristic", "network", "ctg_id"])

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10
    line_idx = 0

    for (heur, net, cid), grp in sorted(groups):
        grp_sorted = grp.sort_values("ratio", ascending=False)
        cumulative = grp_sorted["weighted_contribution"].cumsum().values
        trace_labels = grp_sorted["trace"].values
        x = np.arange(1, len(cumulative) + 1)

        color = cmap(line_idx / max(len(groups) - 1, 1))
        ax.plot(x, cumulative, marker="o", label=f"{heur} {net} ctg{cid}", color=color)

        for xi, yi, lbl in zip(x, cumulative, trace_labels):
            ax.annotate(lbl, (xi, yi), fontsize=5, rotation=30, ha="left", va="bottom")

        line_idx += 1

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.4, label="ratio = 1.0")
    ax.set_xlabel("Trace index (sorted worst → best ratio)")
    ax.set_ylabel("Cumulative weighted ratio")
    ax.set_title("Cumulative Expected Value Build-up per CTG")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_expected_value.pdf")
    plt.close(fig)


def generate_expected_value_charts(
    results: pd.DataFrame, summary: pd.DataFrame, out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if results.empty:
        print("  No data for expected-value charts.")
        return

    plot_stacked_weighted_contributions(results, out_dir)
    plot_bubble_probability_ratio(results, out_dir)
    plot_expected_vs_mean_ratio(results, summary, out_dir)
    plot_cumulative_expected_value(results, out_dir)
    print(f"  Expected-value charts saved to {out_dir}")


# ---------------------------------------------------------------------------
# CTG-level charts (from summary_by_ctg.csv)
# ---------------------------------------------------------------------------

def plot_expected_ratio_by_ctg(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: expected_ratio per CTG, grouped by heuristic."""
    fig, ax = plt.subplots(figsize=(12, 5))
    heuristics = sorted(df["heuristic"].unique())
    ctg_ids = sorted(df["ctg_id"].unique())
    networks = sorted(df["network"].unique())

    x = np.arange(len(ctg_ids) * len(networks))
    width = 0.8 / max(len(heuristics), 1)
    xlabels = []

    for i, heur in enumerate(heuristics):
        vals = []
        for net in networks:
            for ctg_id in ctg_ids:
                row = df[(df["heuristic"] == heur) & (df["network"] == net) & (df["ctg_id"] == ctg_id)]
                vals.append(row["expected_ratio"].values[0] if len(row) else 0)
                if i == 0:
                    xlabels.append(f"{net}\nctg{ctg_id}")
        ax.bar(x + i * width, vals, width, label=heur, edgecolor="black", linewidth=0.3)

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="ratio = 1.0")
    ax.set_xticks(x + width * (len(heuristics) - 1) / 2)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Expected ratio (online/offline)")
    ax.set_title("Expected Makespan Ratio per CTG")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "expected_ratio_by_ctg.pdf")
    plt.close(fig)


def plot_expected_makespan_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bars: expected online vs offline makespan side by side per CTG."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ctg_ids = sorted(df["ctg_id"].unique())
    networks = sorted(df["network"].unique())
    heuristics = sorted(df["heuristic"].unique())

    group_labels = []
    online_vals = []
    offline_vals = []

    for heur in heuristics:
        for net in networks:
            for ctg_id in ctg_ids:
                row = df[(df["heuristic"] == heur) & (df["network"] == net) & (df["ctg_id"] == ctg_id)]
                if len(row):
                    online_vals.append(row["expected_online_makespan"].values[0])
                    offline_vals.append(row["expected_offline_makespan"].values[0])
                    group_labels.append(f"{heur}\n{net}\nctg{ctg_id}")

    x = np.arange(len(group_labels))
    w = 0.35
    ax.bar(x - w / 2, online_vals, w, label="Expected Online (CTG)", edgecolor="black", linewidth=0.3)
    ax.bar(x + w / 2, offline_vals, w, label="Expected Offline (Standalone)", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=7)
    ax.set_ylabel("Expected makespan")
    ax.set_title("Expected Online vs Offline Makespan")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "expected_makespan_comparison.pdf")
    plt.close(fig)


def plot_mean_gap_by_ctg(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: mean gap per CTG grouped by heuristic."""
    fig, ax = plt.subplots(figsize=(12, 5))
    heuristics = sorted(df["heuristic"].unique())
    ctg_ids = sorted(df["ctg_id"].unique())
    networks = sorted(df["network"].unique())

    x = np.arange(len(ctg_ids) * len(networks))
    width = 0.8 / max(len(heuristics), 1)
    xlabels = []

    for i, heur in enumerate(heuristics):
        vals = []
        for net in networks:
            for ctg_id in ctg_ids:
                row = df[(df["heuristic"] == heur) & (df["network"] == net) & (df["ctg_id"] == ctg_id)]
                vals.append(row["mean_gap"].values[0] if len(row) else 0)
                if i == 0:
                    xlabels.append(f"{net}\nctg{ctg_id}")
        ax.bar(x + i * width, vals, width, label=heur, edgecolor="black", linewidth=0.3)

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.4)
    ax.set_xticks(x + width * (len(heuristics) - 1) / 2)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Mean gap (CTG - Standalone)")
    ax.set_title("Mean Trace Gap per CTG")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mean_gap_by_ctg.pdf")
    plt.close(fig)


def plot_expected_ratio_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap: heuristic x (network, ctg_id) with expected_ratio as color."""
    heuristics = sorted(df["heuristic"].unique())
    combos = sorted(df[["network", "ctg_id"]].drop_duplicates().itertuples(index=False))
    combo_labels = [f"{net} ctg{cid}" for net, cid in combos]

    matrix = np.full((len(heuristics), len(combos)), np.nan)
    for i, heur in enumerate(heuristics):
        for j, (net, cid) in enumerate(combos):
            row = df[(df["heuristic"] == heur) & (df["network"] == net) & (df["ctg_id"] == cid)]
            if len(row):
                matrix[i, j] = row["expected_ratio"].values[0]

    fig, ax = plt.subplots(figsize=(max(8, len(combos) * 1.5), max(4, len(heuristics) * 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0.8, vmax=1.5)
    ax.set_xticks(range(len(combo_labels)))
    ax.set_xticklabels(combo_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(heuristics)))
    ax.set_yticklabels(heuristics)

    for i in range(len(heuristics)):
        for j in range(len(combos)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="Expected ratio (online/offline)")
    ax.set_title("Expected Ratio Heatmap: Heuristic × (Network, CTG)")
    fig.tight_layout()
    fig.savefig(out_dir / "expected_ratio_heatmap.pdf")
    plt.close(fig)


def plot_expected_ratio_by_heuristic_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    """Box plot: expected_ratio distribution by heuristic across all CTGs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    heuristics = sorted(df["heuristic"].unique())
    data = [df[df["heuristic"] == h]["expected_ratio"].dropna() for h in heuristics]

    bp = ax.boxplot(data, tick_labels=heuristics, patch_artist=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(heuristics)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Expected ratio (online/offline)")
    ax.set_title("Expected Ratio by Heuristic")
    fig.tight_layout()
    fig.savefig(out_dir / "expected_ratio_by_heuristic_boxplot.pdf")
    plt.close(fig)


def generate_ctg_charts(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("  No CTG-level data to plot.")
        return

    plot_expected_ratio_by_ctg(df, out_dir)
    plot_expected_makespan_comparison(df, out_dir)
    plot_mean_gap_by_ctg(df, out_dir)
    plot_expected_ratio_heatmap(df, out_dir)
    plot_expected_ratio_by_heuristic_boxplot(df, out_dir)
    print(f"  CTG-level charts saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize(run_dirs: List[Path]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    viz_root = thisdir / "visualizations" / ts
    viz_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {len(run_dirs)} run(s) ...")
    results, summary = load_runs(run_dirs)

    print(f"  results.csv rows: {len(results)}")
    print(f"  summary_by_ctg.csv rows: {len(summary)}")

    generate_trace_charts(results, viz_root / "trace_level")
    generate_ctg_charts(summary, viz_root / "ctg_level")
    generate_expected_value_charts(results, summary, viz_root / "expected_value")

    print(f"\nAll visualizations saved to {viz_root}")


def main():
    # Single run
    #visualize([Path("runs/2026-04-29T024010Z_seed44")])

    # Multiple runs (results are concatenated)
    #visualize([
    #    Path("runs/2026-04-29T032545Z_seed47"),
    #    Path("runs/2026-04-29T024010Z_seed44"),
    #])

    # All runs
    visualize([p for p in Path("runs").iterdir() if p.is_dir()])


if __name__ == "__main__":
    main()
