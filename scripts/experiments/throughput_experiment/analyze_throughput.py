import logging
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


thisdir = pathlib.Path(__file__).parent.resolve()
resultsdir = thisdir / "results" / "throughput"
outputdir = thisdir / "output" / "throughput"

SCHEDULER_RENAMES = {
    "Cpop": "CPoP",
    "Heft": "HEFT",
}

_INSPIRIT_BASES = ["Inspirit_HEFT", "Inspirit_CPoP", "Inspirit_FIFO"]
_INSPIRIT_RE = re.compile(r"^(Inspirit_(?:HEFT|CPoP|FIFO))_\d+_\d+$")
_INSPIRIT_BASE_TO_SCHEDULER = {
    "Inspirit_HEFT": "HEFT",
    "Inspirit_CPoP": "CPoP",
    "Inspirit_FIFO": "FIFO",
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


def _parse_workflow(dataset: str) -> str:
    """Extract workflow name from dataset like 'epigenomics_ccr_2.0'."""
    m = re.match(r"^(.+)_ccr_\d+(?:\.\d+)?$", dataset)
    return m.group(1) if m else dataset


def _parse_ccr(dataset: str) -> str:
    """Extract CCR value from dataset like 'epigenomics_ccr_2.0'."""
    m = re.match(r"^.+_ccr_(\d+(?:\.\d+)?)$", dataset)
    return m.group(1) if m else dataset


def _inspirit_base(scheduler: str) -> str | None:
    m = _INSPIRIT_RE.match(scheduler)
    return m.group(1) if m else None


def _add_ratios(data: pd.DataFrame) -> pd.DataFrame:
    """Add per-instance normalized throughput_ratio and makespan_ratio columns."""
    data = data.copy()
    best_tp = data.groupby("Instance")["Throughput"].transform("max")
    best_ms = data.groupby("Instance")["Makespan"].transform("min")
    data["throughput_ratio"] = data["Throughput"] / best_tp
    data["makespan_ratio"] = data["Makespan"] / best_ms
    return data


def _filter_inspirit(group: pd.DataFrame, rank_col: str, higher_is_better: bool) -> pd.DataFrame:
    """
    For each Inspirit base type, keep only the best and worst performing variants
    (by mean of rank_col). Non-Inspirit schedulers pass through unchanged.
    """
    non_insp = group[group["Scheduler"].apply(lambda s: _inspirit_base(s) is None)]
    parts = [non_insp]

    for base in _INSPIRIT_BASES:
        sub = group[group["Scheduler"].apply(lambda s: _inspirit_base(s) == base)]
        if sub.empty:
            continue
        means = sub.groupby("Scheduler")[rank_col].mean()
        if higher_is_better:
            best, worst = means.idxmax(), means.idxmin()
        else:
            best, worst = means.idxmin(), means.idxmax()

        parts.append(sub[sub["Scheduler"] == best])

        if worst != best:
            parts.append(sub[sub["Scheduler"] == worst])

    return pd.concat(parts, ignore_index=True)


def _make_boxplot(
    group: pd.DataFrame,
    metric: str,
    xlabel: str,
    title: str,
    outdir: pathlib.Path,
    filename: str,
) -> None:
    order = (
        group.groupby("Scheduler")[metric]
        .median()
        .sort_values()
        .index.tolist()
    )
    plot_data = [group.loc[group["Scheduler"] == s, metric].values for s in order]

    fig, ax = plt.subplots(figsize=(6, max(4, len(order) * 0.35)))
    ax.boxplot(
        plot_data,
        vert=False,
        patch_artist=True,
        tick_labels=order,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"facecolor": "steelblue", "alpha": 0.7},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.5},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    ax.set_title(title, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=6)
    ax.tick_params(axis="both", labelsize=6)
    fig.tight_layout(pad=0.5)

    safe = str(filename).replace("/", "_")
    fig.savefig(outdir / f"{safe}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _best_inspirit_vs_base(
    group: pd.DataFrame,
    insp_base: str,
    base_sched: str,
    metric: str,
    higher_is_better: bool,
    ccrs: list[str],
) -> tuple[str, list, list] | None:
    """
    Find the best-performing variant of insp_base (by mean of metric, within group)
    and return per-CCR value arrays for it alongside base_sched, restricted to CCRs
    where both have data.
    """
    insp_sub = group[group["Scheduler"].apply(lambda s: _inspirit_base(s) == insp_base)]
    base_sub = group[group["Scheduler"] == base_sched]
    if insp_sub.empty or base_sub.empty:
        return None

    means = insp_sub.groupby("Scheduler")[metric].mean()
    best_variant = means.idxmax() if higher_is_better else means.idxmin()

    insp_by_ccr = [
        insp_sub.loc[(insp_sub["ccr"] == c) & (insp_sub["Scheduler"] == best_variant), metric].values
        for c in ccrs
    ]
    base_by_ccr = [base_sub.loc[base_sub["ccr"] == c, metric].values for c in ccrs]

    keep = [i for i in range(len(ccrs)) if len(insp_by_ccr[i]) > 0 and len(base_by_ccr[i]) > 0]
    if not keep:
        return None

    return (
        best_variant,
        [insp_by_ccr[i] for i in keep],
        [base_by_ccr[i] for i in keep],
        [ccrs[i] for i in keep],
    )


def _make_faceted_grouped_boxplot(
    panels: list[tuple[str, list[str], dict[str, list]]],
    ylabel: str,
    suptitle: str,
    outdir: pathlib.Path,
    filename: str,
) -> None:
    """One figure with one subplot per panel (panel_title, categories, series)."""
    colors = ["steelblue", "darkorange"]

    fig, axes = plt.subplots(1, len(panels), figsize=(max(5, len(panels) * 4), 4), squeeze=False)
    axes = axes[0]

    for ax, (panel_title, categories, series) in zip(axes, panels):
        n_series = len(series)
        width = 0.8 / n_series
        legend_handles = []
        for i, (label, values) in enumerate(series.items()):
            offset = (i - (n_series - 1) / 2) * width
            positions = [j + offset for j in range(len(categories))]
            ax.boxplot(
                values,
                positions=positions,
                widths=width * 0.85,
                patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.0},
                boxprops={"facecolor": colors[i % len(colors)], "alpha": 0.7},
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.5},
                whiskerprops={"linewidth": 0.8},
                capprops={"linewidth": 0.8},
            )
            legend_handles.append(plt.Line2D([0], [0], color=colors[i % len(colors)], lw=6, label=label))

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=7)
        ax.set_xlabel("CCR", fontsize=7)
        ax.set_title(panel_title, fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        ax.legend(handles=legend_handles, fontsize=6, loc="best")

    axes[0].set_ylabel(ylabel, fontsize=7)
    fig.suptitle(suptitle, fontsize=8)
    fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.94))

    safe = str(filename).replace("/", "_")
    fig.savefig(outdir / f"{safe}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _make_heatmap(
    pivot: pd.DataFrame,
    value_label: str,
    title: str,
    outdir: pathlib.Path,
    filename: str,
) -> None:
    fig, ax = plt.subplots(
        figsize=(max(4, len(pivot.columns) * 0.9 + 1.5), max(3, len(pivot.index) * 0.28))
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    ax.set_xlabel("CCR", fontsize=7)

    vals = pivot.values
    finite = vals[~np.isnan(vals)]
    midpoint = (finite.min() + finite.max()) / 2 if finite.size else 0.0
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                     color="white" if v < midpoint else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(value_label, fontsize=6)

    ax.set_title(title, fontsize=8)
    fig.tight_layout(pad=0.5)

    safe = str(filename).replace("/", "_")
    fig.savefig(outdir / f"{safe}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_csv_rankings(data: pd.DataFrame, outpath: pathlib.Path) -> None:
    rows = []

    for dataset, group in data.groupby("dataset"):
        for metric, col, higher in [
            ("Throughput", "throughput_ratio", True),
            ("Makespan", "makespan_ratio", False),
        ]:
            filtered = _filter_inspirit(group, col, higher)
            stats = (
                filtered.groupby("Scheduler")[col]
                .agg(Mean="mean", Median="median", Std="std")
                .reset_index()
            )

            if higher:
                best5 = stats.nlargest(5, "Median")
                worst5 = stats.nsmallest(5, "Median")
            else:
                best5 = stats.nsmallest(5, "Median")
                worst5 = stats.nlargest(5, "Median")

            for rank_type, ranked in [("Best", best5), ("Worst", worst5)]:
                for i, (_, row) in enumerate(ranked.iterrows(), 1):
                    rows.append({
                        "Dataset": dataset,
                        "Metric": metric,
                        "Rank_Type": rank_type,
                        "Rank": i,
                        "Scheduler": row["Scheduler"],
                        "Mean_Ratio": round(row["Mean"], 4),
                        "Median_Ratio": round(row["Median"], 4),
                        "Std_Ratio": round(row["Std"], 4) if not pd.isna(row["Std"]) else 0.0,
                    })

    pd.DataFrame(rows).to_csv(outpath, index=False)


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

    data["Scheduler"] = data["Scheduler"].str.replace("Scheduler", "", regex=False)
    data["Scheduler"] = data["Scheduler"].replace(SCHEDULER_RENAMES)
    data = _add_ratios(data)
    data["workflow"] = data["dataset"].apply(_parse_workflow)
    data["ccr"] = data["dataset"].apply(_parse_ccr)

    # Heatmap: one per workflow, schedulers x CCR, colored by median throughput ratio.
    # Replaces a separate boxplot per (workflow, CCR) with a single compact summary.
    heatmap_dir = outputdir / "heatmap"
    heatmap_dir.mkdir(exist_ok=True)
    for workflow, wf_group in data.groupby("workflow"):
        filtered = _filter_inspirit(wf_group, "throughput_ratio", higher_is_better=True)
        pivot = filtered.pivot_table(index="Scheduler", columns="ccr", values="throughput_ratio", aggfunc="median")
        pivot = pivot[sorted(pivot.columns, key=float)]
        pivot = pivot.loc[pivot.mean(axis=1, skipna=True).sort_values(ascending=False).index]

        hm_title = f"{title} — {workflow}" if title else str(workflow)
        _make_heatmap(
            pivot,
            value_label="Throughput ratio vs best (1.0 = best)",
            title=hm_title,
            outdir=heatmap_dir,
            filename=workflow,
        )

    # Workflow throughput graphs: combine all CCRs, Inspirit filtered to best/worst
    workflow_dir = outputdir / "workflow"
    workflow_dir.mkdir(exist_ok=True)
    for workflow, wf_group in data.groupby("workflow"):
        filtered = _filter_inspirit(wf_group, "throughput_ratio", higher_is_better=True)
        wf_title = f"{title} — {workflow} (all CCR)" if title else f"{workflow} (all CCR)"
        _make_boxplot(
            filtered,
            metric="throughput_ratio",
            xlabel="Throughput ratio vs best (1.0 = best)",
            title=wf_title,
            outdir=workflow_dir,
            filename=workflow,
        )

    # CSV rankings: best/worst 5 per dataset for throughput and makespan
    csv_dir = outputdir / "csv"
    csv_dir.mkdir(exist_ok=True)
    _write_csv_rankings(data, csv_dir / "rankings.csv")

    # Base comparison graphs: best Inspirit variant vs its base scheduler, grouped by CCR.
    # Faceted one subplot per Inspirit type (HEFT/CPoP/FIFO) so one file covers all three.
    base_comp_dir = outputdir / "base_comparison"
    base_comp_dir.mkdir(exist_ok=True)

    for metric, ylabel, higher in [
        ("throughput_ratio", "Throughput ratio vs best (1.0 = best)", True),
        ("makespan_ratio", "Makespan ratio vs best (1.0 = best)", False),
    ]:
        for workflow, wf_group in data.groupby("workflow"):
            ccrs = sorted(wf_group["ccr"].unique(), key=float)
            panels = []
            for insp_base, base_sched in _INSPIRIT_BASE_TO_SCHEDULER.items():
                result = _best_inspirit_vs_base(wf_group, insp_base, base_sched, metric, higher, ccrs)
                if result is None:
                    continue
                best_variant, insp_data, base_data, kept_ccrs = result
                panels.append((
                    f"{best_variant} vs {base_sched}",
                    kept_ccrs,
                    {best_variant: insp_data, base_sched: base_data},
                ))
            if not panels:
                continue
            suptitle = f"{title} — {workflow}" if title else str(workflow)
            _make_faceted_grouped_boxplot(
                panels=panels,
                ylabel=ylabel,
                suptitle=suptitle,
                outdir=base_comp_dir,
                filename=f"{workflow}_{metric}",
            )

        # Overall: combine all workflows for each Inspirit type
        ccrs_all = sorted(data["ccr"].unique(), key=float)
        panels = []
        for insp_base, base_sched in _INSPIRIT_BASE_TO_SCHEDULER.items():
            result = _best_inspirit_vs_base(data, insp_base, base_sched, metric, higher, ccrs_all)
            if result is None:
                continue
            best_variant, insp_data, base_data, kept_ccrs = result
            panels.append((
                f"{best_variant} vs {base_sched}",
                kept_ccrs,
                {best_variant: insp_data, base_sched: base_data},
            ))
        if not panels:
            continue
        suptitle = f"{title} — Overall" if title else "Overall"
        _make_faceted_grouped_boxplot(
            panels=panels,
            ylabel=ylabel,
            suptitle=suptitle,
            outdir=base_comp_dir,
            filename=f"overall_{metric}",
        )


def main():
    run_analysis(
        resultsdir=resultsdir,
        outputdir=outputdir,
        title="Throughput Benchmarking",
    )


if __name__ == "__main__":
    main()
