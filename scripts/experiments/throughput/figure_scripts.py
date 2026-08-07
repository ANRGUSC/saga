"""Generate publication-ready figures/tables for the throughput experiment.

Each function renders one output (currently as a compiled LaTeX PDF, via
pdflatex + pdfcrop) into output/figures/.

Usage:
    python figure_scripts.py throughput_ratio_table_wfcommons throughput_ratio_table_riotbench
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from common import outputdir, resultsdir

# Figures are emitted as PDF: vector output scales to the column width without
# resampling, and the labels stay real text in the typeset paper. fonttype 42
# embeds TrueType rather than Type 3, which some publishers reject.
FIG_EXT = "pdf"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


INSTANCE_KEYS = ["Branch", "Regime", "Workflow", "CCR", "Instance", "Seed"]

RESULT_FILES = [
    ("riotbench", "deterministic"),
    ("riotbench", "stochastic"),
    ("wfcommons", "deterministic"),
    ("wfcommons", "stochastic"),
]

_TP_SCHEDULER_RE = re.compile(r"^(HEFT|CPoP)-Tp_(\w+)$")

_ALGO_ORDER = ["HEFT", "CPoP"]
_POLICY_ORDER = [
    "static", "reschedule", "conditional",
    "random50", "random25", "random10", "random5", "random1",
    "checkpoint_10", "checkpoint_quarterly", "checkpoint_mid",
]
_POLICY_LABEL = {
    "static": "Static",
    "reschedule": "Reschedule",
    "conditional": "Conditional",
    "random50": r"Random-50\%",
    "random25": r"Random-25\%",
    "random10": r"Random-10\%",
    "random5": r"Random-5\%",
    "random1": r"Random-1\%",
    "checkpoint_10": r"Checkpoint-10\%",
    "checkpoint_quarterly": "Checkpoint-Qtr",
    "checkpoint_mid": "Checkpoint-Mid",
}
# Plain-text (non-LaTeX-escaped) labels, for matplotlib figures.
_POLICY_LABEL_PLAIN = {k: v.replace(r"\%", "%") for k, v in _POLICY_LABEL.items()}
_EXEC_TYPE_ORDER = ["Deterministic", "Stochastic"]

figuresdir = outputdir / "figures"
# Subfolders, one per figure "family" - keeps the ~50+ output files navigable.
tables_dir = figuresdir / "tables"
reschedule_boxplots_dir = figuresdir / "reschedule_boxplots"
throughput_boxplots_dir = reschedule_boxplots_dir / "throughput"
makespan_boxplots_dir = reschedule_boxplots_dir / "makespan"
highlight_boxplots_dir = reschedule_boxplots_dir / "highlight"
relative_boxplots_dir = reschedule_boxplots_dir / "relative"
all_policy_boxplots_dir = reschedule_boxplots_dir / "all_policy"
base_static_vs_tp_reschedule_dir = figuresdir / "base_static_vs_tp_reschedule"
tp_static_vs_tp_reschedule_dir = figuresdir / "tp_static_vs_tp_reschedule"
all_baselines_vs_best_dir = figuresdir / "all_baselines_vs_best"
reschedule_improvement_plot_dir = figuresdir / "reschedule_improvement_plot"


def _load_results(branch: str) -> pd.DataFrame:
    frames = []
    for file_branch, regime in RESULT_FILES:
        if file_branch != branch:
            continue
        path = resultsdir / f"{file_branch}_{regime}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"No result CSVs found for branch '{branch}' in {resultsdir}")
    return pd.concat(frames, ignore_index=True)


def _throughput_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Instance-level TR = Throughput_scheduler / Throughput_base for every
    TP-HEFT/TP-CPoP scheduler variant, matched to its non-TP base scheduler on
    the same problem instance."""
    tp_schedulers = sorted(s for s in df["Scheduler"].unique() if _TP_SCHEDULER_RE.match(s))
    rows = []
    for tp_name in tp_schedulers:
        algo, policy = _TP_SCHEDULER_RE.match(tp_name).groups()
        base_name = f"{algo}_{policy}"
        tp_df = df[df["Scheduler"] == tp_name]
        base_df = df[df["Scheduler"] == base_name]
        if base_df.empty:
            continue
        merged = tp_df.merge(
            base_df[INSTANCE_KEYS + ["Throughput"]],
            on=INSTANCE_KEYS,
            suffixes=("", "_base"),
        )
        merged["Algo"] = algo
        merged["Policy"] = policy
        merged["SchedulerName"] = f"T-{algo}-{_POLICY_LABEL[policy]}"
        merged["BaseName"] = f"{algo}-{_POLICY_LABEL[policy]}"
        merged["ExecutionType"] = merged["Regime"].str.capitalize()
        merged["TR"] = merged["Throughput"] / merged["Throughput_base"]
        rows.append(merged[["Algo", "Policy", "SchedulerName", "BaseName", "ExecutionType", "TR"]])
    return pd.concat(rows, ignore_index=True)


def _summarize(tr: pd.Series) -> pd.Series:
    """Geometric mean (the only arithmetic-mean-safe way to average a ratio -
    it's invariant to which scheduler is the numerator, unlike the arithmetic
    mean) plus the interquartile range as a robust spread that isn't blown up
    by rare outlier instances."""
    values = tr.to_numpy()
    geomean = float(np.exp(np.mean(np.log(values))))
    p25, p75 = np.percentile(values, [25, 75])
    return pd.Series({"geomean": geomean, "p25": p25, "p75": p75})


def _aggregate(ratios: pd.DataFrame) -> pd.DataFrame:
    agg = (
        ratios.groupby(["Algo", "Policy", "SchedulerName", "BaseName", "ExecutionType"])["TR"]
        .apply(_summarize)
        .unstack()
        .reset_index()
    )
    agg["_exec_rank"] = agg["ExecutionType"].map(_EXEC_TYPE_ORDER.index)
    agg["_algo_rank"] = agg["Algo"].map(_ALGO_ORDER.index)
    agg["_policy_rank"] = agg["Policy"].map(_POLICY_ORDER.index)
    agg = agg.sort_values(["_exec_rank", "_algo_rank", "_policy_rank"]).reset_index(drop=True)
    return agg


def _escape_tex(s: str) -> str:
    return s.replace("_", r"\_")


def _render_latex(agg: pd.DataFrame, caption: str) -> str:
    lines = [
        r"\documentclass[border=6pt]{standalone}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\begin{document}",
        r"\begin{tabular}{@{}llc@{}}",
        r"\multicolumn{3}{c}{\textbf{" + _escape_tex(caption) + r"}} \\",
        r"\toprule",
        r"Scheduler & Base & $TR_{\mathrm{Scheduler,Base}}$ (geomean [IQR]) \\",
        r"\midrule",
    ]
    prev_group = None
    for _, row in agg.iterrows():
        group = (row["ExecutionType"], row["Algo"])
        if prev_group is not None and group != prev_group:
            lines.append(r"\midrule")
        prev_group = group
        lines.append(
            f"{_escape_tex(row['SchedulerName'])} & {_escape_tex(row['BaseName'])} & "
            f"{row['geomean']:.3f} [{row['p25']:.3f}, {row['p75']:.3f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{document}"]
    return "\n".join(lines)


def _compile_pdf(tex_source: str, name: str) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tex_file = tmp_path / f"{name}.tex"
        tex_file.write_text(tex_source)
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file.name],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        pdf_file = tmp_path / f"{name}.pdf"
        if result.returncode != 0 or not pdf_file.exists():
            log_tail = "\n".join(result.stdout.splitlines()[-40:])
            raise RuntimeError(f"pdflatex failed for {name}:\n{log_tail}")

        cropped = tmp_path / f"{name}-crop.pdf"
        crop_result = subprocess.run(
            ["pdfcrop", pdf_file.name, cropped.name],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        final_pdf = cropped if crop_result.returncode == 0 and cropped.exists() else pdf_file

        dest_tex = tables_dir / f"{name}.tex"
        dest_pdf = tables_dir / f"{name}.pdf"
        shutil.copy(tex_file, dest_tex)
        shutil.copy(final_pdf, dest_pdf)
    return dest_pdf


_BRANCH_LABEL = {"wfcommons": "WfCommons", "riotbench": "RIoTBench"}

_RESCHEDULE_POLICIES = [
    "reschedule", "conditional", "random50", "random25", "random10", "random5", "random1",
    "checkpoint_10", "checkpoint_quarterly", "checkpoint_mid",
]

_SCHEDULER_RE = re.compile(r"^(HEFT-Tp|CPoP-Tp|HEFT|CPoP|MaxTP)_(\w+)$")


def _refire_pct_rows() -> pd.DataFrame:
    """Row-level reschedule-firing-rate: RefirePct = 100 * RescheduleCount /
    RescheduleCount_reschedule, matched per (Branch, Regime, Workflow, CCR,
    Instance, Seed, Algo). The "reschedule" policy reschedules after every
    task completion, so it's the natural 100% reference; every other
    policy's count is expressed relative to it on the same (instance, algo)
    pair."""
    frames = [_load_results(branch) for branch in _BRANCH_LABEL]
    df = pd.concat(frames, ignore_index=True)
    df = df.copy()
    df[["Algo", "Policy"]] = df["Scheduler"].str.extract(_SCHEDULER_RE)
    df = df[df["Policy"].isin(_RESCHEDULE_POLICIES)]

    join_keys = ["Branch", "Regime", "Workflow", "CCR", "Instance", "Seed", "Algo"]
    reference = df.loc[df["Policy"] == "reschedule", join_keys + ["RescheduleCount"]]
    merged = df.merge(reference, on=join_keys, suffixes=("", "_ref"))
    merged["RefirePct"] = 100 * merged["RescheduleCount"] / merged["RescheduleCount_ref"]
    return merged


def _policy_fire_order() -> list:
    """Policy names ordered by mean firing rate (pooled across branch/algo),
    ascending - the x-axis order for the improvement plot."""
    merged = _refire_pct_rows()
    return merged.groupby("Policy")["RefirePct"].mean().sort_values().index.tolist()


def _reschedule_stats() -> pd.DataFrame:
    """Reschedule-firing-rate statistics per (Branch, Policy), pooled across
    every scheduler/algo variant (HEFT, CPoP, their -Tp counterparts, and MaxTP
    all fire their policy the same way, so splitting by scheduler adds no
    information)."""
    merged = _refire_pct_rows()
    agg = (
        merged.groupby(["Branch", "Policy"])["RefirePct"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["_branch_rank"] = agg["Branch"].map(list(_BRANCH_LABEL).index)
    agg["_policy_rank"] = agg["Policy"].map(_POLICY_ORDER.index)
    agg = agg.sort_values(["_branch_rank", "_policy_rank"]).reset_index(drop=True)
    return agg


def _reschedule_count_tr_by_ccr() -> pd.DataFrame:
    """Mean TR = Throughput_policy / Throughput_TP-static per (Branch,
    Policy, CCR), pivoted so each CCR is its own column - TP-HEFT, TP-CPoP,
    and MaxTP pooled together (MaxTP's static baseline is the bare "MaxTP"
    scheduler name, not "MaxTP_static" which is never generated - see
    _maxtp_rows)."""
    tp_rows = _improvement_rows("Throughput")
    tp_rows = tp_rows[tp_rows["Variant"] == "T"]
    maxtp_rows = _maxtp_rows("Throughput")
    rows = pd.concat([tp_rows, maxtp_rows], ignore_index=True)
    agg = rows.groupby(["Branch", "Policy", "CCR"])["TR"].mean().reset_index()
    pivot = agg.pivot(index=["Branch", "Policy"], columns="CCR", values="TR").reset_index()
    return pivot


def _reschedule_count_stats() -> pd.DataFrame:
    """_reschedule_stats (reschedule-firing-rate) joined with
    _reschedule_count_tr_by_ccr (per-CCR TR vs TP-static) on (Branch, Policy)."""
    rate = _reschedule_stats()
    tr_by_ccr = _reschedule_count_tr_by_ccr()
    return rate.merge(tr_by_ccr, on=["Branch", "Policy"], how="left")


def _render_reschedule_latex(agg: pd.DataFrame, caption: str) -> str:
    n_ccr = len(_CCR_ORDER)
    col_spec = "ll" + "c" * n_ccr + "cc"
    ccr_header = " & ".join(f"{ccr:g}" for ccr in _CCR_ORDER)
    lines = [
        r"\documentclass[border=6pt]{standalone}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\begin{document}",
        r"\begin{tabular}{@{}" + col_spec + r"@{}}",
        r"\multicolumn{" + str(len(col_spec)) + r"}{c}{\textbf{" + _escape_tex(caption) + r"}} \\",
        r"\toprule",
        r" & & \multicolumn{" + str(n_ccr) + r"}{c}{$TR_{\mathrm{Policy,TP\text{-}Static}}$ (mean) by CCR} & & \\",
        r"\cmidrule(lr){3-" + str(2 + n_ccr) + r"}",
        r"Branch & Policy & " + ccr_header + r" & Reschedule Rate vs.\ Reschedule (mean $\pm$ std) & $n$ \\",
        r"\midrule",
    ]
    prev_branch = None
    for _, row in agg.iterrows():
        if prev_branch is not None and row["Branch"] != prev_branch:
            lines.append(r"\midrule")
        prev_branch = row["Branch"]
        ccr_cells = " & ".join(
            f"{row[ccr]:.3f}" if pd.notna(row[ccr]) else "-" for ccr in _CCR_ORDER
        )
        lines.append(
            f"{_BRANCH_LABEL[row['Branch']]} & {_POLICY_LABEL[row['Policy']]} & {ccr_cells} & "
            f"{row['mean']:.1f}\\% $\\pm$ {row['std']:.1f}\\% & {int(row['count'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{document}"]
    return "\n".join(lines)


def reschedule_count_table() -> Path:
    """Table of how often each dynamic reschedule policy (conditional,
    random10/25/50) fired relative to the "reschedule" policy (which
    reschedules after every task and so is always the 100% reference),
    pooled across all schedulers, split by benchmark suite; plus, for each
    CCR, the mean throughput improvement of that policy over its TP-static
    baseline (TP-HEFT, TP-CPoP, and MaxTP pooled together)."""
    agg = _reschedule_count_stats()
    caption = "Table: Reschedule Rate and T-Static Throughput Improvement statistics."
    tex = _render_reschedule_latex(agg, caption=caption)
    name = "reschedule_count_table"
    pdf_path = _compile_pdf(tex, name)
    print(f"{name} -> {pdf_path}")
    return pdf_path


_VARIANT_ORDER = ["Base", "T"]


def _improvement_rows(metric: str = "Throughput") -> pd.DataFrame:
    """Per-instance TR = <metric>_policy / <metric>_static for every dynamic
    reschedule policy against its own algo's static baseline (same
    Branch/Regime/Workflow/CCR/Instance/Seed/Algo). Row-level (CCR kept, not
    aggregated) so callers can facet/box-plot by CCR.

    Base/TP only (MaxTP excluded): this is a HEFT/CPoP-vs-their-Tp-counterpart
    comparison, and MaxTP has no non-TP counterpart to pair it with."""
    frames = [_load_results(branch) for branch in _BRANCH_LABEL]
    df = pd.concat(frames, ignore_index=True)
    df = df.copy()
    df[["Algo", "Policy"]] = df["Scheduler"].str.extract(_SCHEDULER_RE)
    df = df[df["Algo"] != "MaxTP"]

    join_keys = ["Branch", "Regime", "Workflow", "CCR", "Instance", "Seed", "Algo"]
    static = df.loc[df["Policy"] == "static", join_keys + [metric]]
    dynamic = df[df["Policy"].isin(_RESCHEDULE_POLICIES)]
    merged = dynamic.merge(static, on=join_keys, suffixes=("", "_static"))
    merged["TR"] = merged[metric] / merged[f"{metric}_static"]
    merged["Variant"] = merged["Algo"].str.endswith("-Tp").map({True: "T", False: "Base"})
    return merged


def _maxtp_rows(metric: str = "Throughput") -> pd.DataFrame:
    """Same shape as _improvement_rows, but for MaxTP only (Variant "T" -
    MaxTP is inherently throughput-based, like HEFT-Tp/CPoP-Tp, and has no
    non-TP counterpart). Kept separate from _improvement_rows because that
    function backs the Base-vs-TP comparison tables where MaxTP doesn't
    belong; callers that want MaxTP pooled into "T" (e.g. the WfCommons T
    boxplots) concat this in explicitly.

    MaxTP's static baseline is the bare "MaxTP" scheduler name, not
    "MaxTP_static" (which is never generated - see _branch_ccr_stats)."""
    frames = [_load_results(branch) for branch in _BRANCH_LABEL]
    df = pd.concat(frames, ignore_index=True)
    df = df.copy()
    df[["Algo", "Policy"]] = df["Scheduler"].str.extract(_SCHEDULER_RE)
    bare_maxtp = df["Scheduler"] == "MaxTP"
    df.loc[bare_maxtp, "Algo"] = "MaxTP"
    df.loc[bare_maxtp, "Policy"] = "static"
    df = df[df["Algo"] == "MaxTP"]

    join_keys = ["Branch", "Regime", "Workflow", "CCR", "Instance", "Seed", "Algo"]
    static = df.loc[df["Policy"] == "static", join_keys + [metric]]
    dynamic = df[df["Policy"].isin(_RESCHEDULE_POLICIES)]
    merged = dynamic.merge(static, on=join_keys, suffixes=("", "_static"))
    merged["TR"] = merged[metric] / merged[f"{metric}_static"]
    merged["Variant"] = "T"
    return merged


def _improvement_stats() -> pd.DataFrame:
    """TR statistics per (Branch, Variant, Policy) - split by branch and by
    whether the algo is a -Tp variant or a plain (base) comparator,
    rescheduling's effect turns out to differ substantially between the two,
    so pooling them together hides that."""
    merged = _improvement_rows()
    agg = (
        merged.groupby(["Branch", "Variant", "Policy"])["TR"]
        .apply(_summarize)
        .unstack()
        .reset_index()
    )
    agg["_branch_rank"] = agg["Branch"].map(list(_BRANCH_LABEL).index)
    agg["_variant_rank"] = agg["Variant"].map(_VARIANT_ORDER.index)
    agg["_policy_rank"] = agg["Policy"].map(_POLICY_ORDER.index)
    agg = agg.sort_values(["_branch_rank", "_variant_rank", "_policy_rank"]).reset_index(drop=True)
    return agg


def _render_improvement_latex(agg: pd.DataFrame, caption: str) -> str:
    lines = [
        r"\documentclass[border=6pt]{standalone}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\begin{document}",
        r"\begin{tabular}{@{}lllc@{}}",
        r"\multicolumn{4}{c}{\textbf{" + _escape_tex(caption) + r"}} \\",
        r"\toprule",
        r"Branch & Variant & Policy & $TR_{\mathrm{Policy,Static}}$ (geomean [IQR]) \\",
        r"\midrule",
    ]
    prev_group = None
    for _, row in agg.iterrows():
        group = (row["Branch"], row["Variant"])
        if prev_group is not None and group != prev_group:
            lines.append(r"\midrule")
        prev_group = group
        lines.append(
            f"{_BRANCH_LABEL[row['Branch']]} & {row['Variant']} & {_POLICY_LABEL[row['Policy']]} & "
            f"{row['geomean']:.3f} [{row['p25']:.3f}, {row['p75']:.3f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{document}"]
    return "\n".join(lines)


def reschedule_improvement_table() -> Path:
    """Table of the relative throughput improvement of each dynamic
    reschedule policy over its own algo's static baseline: geometric mean
    (and IQR) of Throughput_policy/Throughput_static per instance, pooled
    across algo variants, split by benchmark suite."""
    agg = _improvement_stats()
    caption = "Table: Reschedule Policy Improvement over Static."
    tex = _render_improvement_latex(agg, caption=caption)
    name = "reschedule_improvement_table"
    pdf_path = _compile_pdf(tex, name)
    print(f"{name} -> {pdf_path}")
    return pdf_path


# reschedule (fires every task completion), conditional (fires on outlier deviations),
# and the three checkpoint cadences - the policies of interest for the WfCommons/TP CCR
# breakdown, in the same relative order as _POLICY_ORDER.
_CCR_TABLE_POLICIES = ["reschedule", "conditional", "checkpoint_10", "checkpoint_quarterly", "checkpoint_mid"]


def _wfcommons_tp_ccr_stats() -> pd.DataFrame:
    """TR statistics per (CCR, Policy), same underlying TR as
    _improvement_stats (Throughput_policy/Throughput_static per instance),
    but scoped to WfCommons TP-variant algos and the reschedule/conditional/
    checkpoint-* policies, and split by CCR instead of pooled across it."""
    merged = _improvement_rows()
    merged = merged[
        (merged["Branch"] == "wfcommons")
        & (merged["Variant"] == "T")
        & (merged["Policy"].isin(_CCR_TABLE_POLICIES))
    ]
    agg = (
        merged.groupby(["CCR", "Policy"])["TR"]
        .apply(_summarize)
        .unstack()
        .reset_index()
    )
    agg["_ccr_rank"] = agg["CCR"].map(_CCR_ORDER.index)
    agg["_policy_rank"] = agg["Policy"].map(_POLICY_ORDER.index)
    agg = agg.sort_values(["_ccr_rank", "_policy_rank"]).reset_index(drop=True)
    return agg


def _render_ccr_improvement_latex(agg: pd.DataFrame, caption: str) -> str:
    lines = [
        r"\documentclass[border=6pt]{standalone}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\begin{document}",
        r"\begin{tabular}{@{}llc@{}}",
        r"\multicolumn{3}{c}{\textbf{" + _escape_tex(caption) + r"}} \\",
        r"\toprule",
        r"CCR & Policy & $TR_{\mathrm{Policy,Static}}$ (geomean [IQR]) \\",
        r"\midrule",
    ]
    prev_ccr = None
    for _, row in agg.iterrows():
        if prev_ccr is not None and row["CCR"] != prev_ccr:
            lines.append(r"\midrule")
        prev_ccr = row["CCR"]
        lines.append(
            f"{row['CCR']:g} & {_POLICY_LABEL[row['Policy']]} & "
            f"{row['geomean']:.3f} [{row['p25']:.3f}, {row['p75']:.3f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{document}"]
    return "\n".join(lines)


def wfcommons_tp_ccr_improvement_table() -> Path:
    """Table of the relative throughput improvement of reschedule/conditional/
    checkpoint-* policies over their algo's static baseline, for WfCommons
    TP-variant algos (HEFT-Tp, CPoP-Tp) only, split out per CCR instead of
    pooled across it - the same underlying statistic as
    reschedule_improvement_table, narrowed to this policy/branch/variant
    subset and broken out by CCR."""
    agg = _wfcommons_tp_ccr_stats()
    caption = "Table: WfCommons T Reschedule Policy Improvement over Static, by CCR."
    tex = _render_ccr_improvement_latex(agg, caption=caption)
    name = "wfcommons_tp_ccr_improvement_table"
    pdf_path = _compile_pdf(tex, name)
    print(f"{name} -> {pdf_path}")
    return pdf_path


# Base = plain EFT comparator (HEFT, CPoP); TP = throughput-bottleneck comparator
# (HEFT-Tp, CPoP-Tp, and MaxTP, which is inherently throughput-based and has no
# separate -Tp counterpart).
_CCR_OVERVIEW_VARIANT_ALGOS = {"Base": ["HEFT", "CPoP"], "T": ["HEFT-Tp", "CPoP-Tp", "MaxTP"]}
_CCR_OVERVIEW_VARIANT_ORDER = list(_CCR_OVERVIEW_VARIANT_ALGOS)


def _branch_ccr_stats() -> pd.DataFrame:
    """TR statistics per (Branch, Variant, CCR), "reschedule" policy only
    (reschedules after every task completion) pooled across every workflow.
    Variant is Base (HEFT, CPoP) or TP (HEFT-Tp, CPoP-Tp, MaxTP) - a
    high-level view of how much rescheduling helps at each CCR, split by
    whether the algo already uses the throughput-bottleneck comparator, with
    no workflow-level detail.

    Built directly (not via _improvement_rows, which drops MaxTP for the
    reschedule_improvement_table's Base-vs-TP comparison where it doesn't
    belong).

    MaxTP's static baseline is the bare "MaxTP" scheduler name, not
    "MaxTP_static" (which is never generated - run.py's standalone
    MaxTPScheduler under the plain "MaxTP" name is already that config, one
    row per Workflow/CCR/Instance/Seed with RescheduleCount == 0)."""
    frames = [_load_results(branch) for branch in _BRANCH_LABEL]
    df = pd.concat(frames, ignore_index=True)
    df = df.copy()
    df[["Algo", "Policy"]] = df["Scheduler"].str.extract(_SCHEDULER_RE)
    bare_maxtp = df["Scheduler"] == "MaxTP"
    df.loc[bare_maxtp, "Algo"] = "MaxTP"
    df.loc[bare_maxtp, "Policy"] = "static"

    algo_variant = {
        algo: variant for variant, algos in _CCR_OVERVIEW_VARIANT_ALGOS.items() for algo in algos
    }
    df["Variant"] = df["Algo"].map(algo_variant)
    df = df[df["Variant"].notna()]

    join_keys = ["Branch", "Regime", "Workflow", "CCR", "Instance", "Seed", "Algo"]
    static = df.loc[df["Policy"] == "static", join_keys + ["Throughput"]]
    dynamic = df[df["Policy"] == "reschedule"]
    merged = dynamic.merge(static, on=join_keys, suffixes=("", "_static"))
    merged["TR"] = merged["Throughput"] / merged["Throughput_static"]

    agg = (
        merged.groupby(["Branch", "Variant", "CCR"])["TR"]
        .apply(_summarize)
        .unstack()
        .reset_index()
    )
    agg["_branch_rank"] = agg["Branch"].map(list(_BRANCH_LABEL).index)
    agg["_variant_rank"] = agg["Variant"].map(_CCR_OVERVIEW_VARIANT_ORDER.index)
    agg["_ccr_rank"] = agg["CCR"].map(_CCR_ORDER.index)
    agg = agg.sort_values(["_branch_rank", "_variant_rank", "_ccr_rank"]).reset_index(drop=True)
    return agg


def _render_branch_ccr_latex(agg: pd.DataFrame, caption: str) -> str:
    lines = [
        r"\documentclass[border=6pt]{standalone}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\begin{document}",
        r"\begin{tabular}{@{}lllc@{}}",
        r"\multicolumn{4}{c}{\textbf{" + _escape_tex(caption) + r"}} \\",
        r"\toprule",
        r"Branch & Variant & CCR & $TR_{\mathrm{Dynamic,Static}}$ (geomean [IQR]) \\",
        r"\midrule",
    ]
    prev_group = None
    for _, row in agg.iterrows():
        group = (row["Branch"], row["Variant"])
        if prev_group is not None and group != prev_group:
            lines.append(r"\midrule")
        prev_group = group
        lines.append(
            f"{_BRANCH_LABEL[row['Branch']]} & {row['Variant']} & {row['CCR']:g} & "
            f"{row['geomean']:.3f} [{row['p25']:.3f}, {row['p75']:.3f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{document}"]
    return "\n".join(lines)


def rescheduling_ccr_overview_table() -> Path:
    """High-level table: how much does the "reschedule" policy improve
    throughput over static at each CCR - pooled across workflow, split by
    Branch and by Variant (Base: HEFT/CPoP vs TP: HEFT-Tp/CPoP-Tp/MaxTP).
    Meant to show the big picture (rescheduling matters more at low CCR,
    and more so for the EFT-based Base comparator than the already
    throughput-aware TP comparator) without workflow-level detail."""
    agg = _branch_ccr_stats()
    caption = "Table: Reschedule Policy Improvement over Static by CCR."
    tex = _render_branch_ccr_latex(agg, caption=caption)
    name = "rescheduling_ccr_overview_table"
    pdf_path = _compile_pdf(tex, name)
    print(f"{name} -> {pdf_path}")
    return pdf_path


# Branch -> hue (validated CVD-safe pair), Variant -> line style (redundant
# encoding so identity never rests on color alone).
_BRANCH_COLOR = {"wfcommons": "#2a78d6", "riotbench": "#eb6834"}
_VARIANT_STYLE = {
    "Base": {"linestyle": "-", "marker": "o"},
    "T": {"linestyle": "--", "marker": "s"},
}


def reschedule_improvement_plot() -> Path:
    """Line plot of relative throughput improvement (geomean TR - 1, i.e. how
    much greater than 1 the policy/static throughput ratio is) for every
    dynamic reschedule policy, split by branch and TP/base variant. Policies
    are ordered left-to-right by how often they actually fire (mean
    RefirePct, pooled) - essentially the reschedule_improvement_table,
    plotted."""
    agg = _improvement_stats()
    policy_order = _policy_fire_order()
    x_pos = {policy: i for i, policy in enumerate(policy_order)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for (branch, variant), group in agg.groupby(["Branch", "Variant"]):
        group = group.set_index("Policy").reindex(policy_order).dropna(how="all")
        xs = [x_pos[policy] for policy in group.index]
        ys = (group["geomean"] - 1) * 100
        ax.plot(
            xs, ys,
            color=_BRANCH_COLOR[branch], label=f"{_BRANCH_LABEL[branch]} ({variant})",
            linewidth=2, markersize=7, **_VARIANT_STYLE[variant],
        )

    ax.axhline(0, color="#898781", linewidth=1, zorder=0)
    ax.set_xticks(range(len(policy_order)))
    ax.set_xticklabels([_POLICY_LABEL_PLAIN[p] for p in policy_order], rotation=20, ha="right")
    ax.set_xlabel("Reschedule policy (ordered by mean firing rate)")
    ax.set_ylabel("Throughput improvement over Static (%)")
    ax.set_title("Relative throughput gain from rescheduling")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=-1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()

    reschedule_improvement_plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = reschedule_improvement_plot_dir / f"reschedule_improvement_plot.{FIG_EXT}"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"reschedule_improvement_plot -> {out_path}")
    return out_path


# Fixed categorical order (heaviest rescheduling -> lightest), each policy's
# own hue throughout. The first 8 are the validated default palette (its full
# 8-slot run); checkpoint_quarterly/checkpoint_mid go past that palette's
# validated slot count, so their hues (teal, brown) were chosen and re-checked
# with the skill's validate_palette.js against this exact 10-slot order --
# all gates pass (CVD sep, normal-vision floor, chroma, lightness); the
# magenta/yellow/aqua sub-3:1-contrast relief warning already applied to the
# original 7 and is unchanged.
_POLICY_COLOR = {
    "reschedule": "#2a78d6",            # blue
    "conditional": "#008300",           # green
    "random50": "#e87ba4",              # magenta
    "random25": "#eda100",              # yellow
    "random10": "#1baf7a",              # aqua
    "random5": "#eb6834",               # orange
    "random1": "#4a3aa7",               # violet
    "checkpoint_10": "#e34948",         # red
    "checkpoint_quarterly": "#0089a3",  # teal
    "checkpoint_mid": "#a3651a",        # brown
}
_BOXPLOT_POLICY_ORDER = [
    "reschedule", "conditional", "random50", "random25", "random10", "random5", "random1",
    "checkpoint_10", "checkpoint_quarterly", "checkpoint_mid",
]

# Test variant that also includes "static" (gray - neutral/no-policy, distinct from the
# 10 saturated dynamic-policy hues above, not part of the validated palette run).
_ALL_POLICY_ORDER = ["static"] + _BOXPLOT_POLICY_ORDER
_ALL_POLICY_COLOR = {**_POLICY_COLOR, "static": "#6b6b6b"}

_CCR_ORDER = [0.2, 0.5, 1.0, 2.0, 5.0]


def _draw_grouped_boxplot(
    ax, data: pd.DataFrame, title: str, y_label: str = "Throughput Ratio",
    policy_order: Optional[list] = None, policy_color: Optional[dict] = None,
) -> None:
    """Grouped box-and-whisker: one box per policy within each CCR group.
    `data` has columns CCR, Policy, TR. Box = IQR, whiskers = 5th/95th
    percentile, median line, outliers beyond that hidden. Percentile
    whiskers (not the standard 1.5xIQR Tukey rule) because a lot of these
    groups are >75% exact ties (TR==1) - Tukey whiskers collapse to zero
    right along with a zero IQR, hiding real spread that's still there in
    the untied minority; percentile whiskers don't have that failure mode.

    policy_order/policy_color default to _BOXPLOT_POLICY_ORDER/_POLICY_COLOR
    (the dynamic-only reschedule policies); pass overrides to plot a
    different policy set (e.g. including "static")."""
    policy_order = policy_order if policy_order is not None else _BOXPLOT_POLICY_ORDER
    policy_color = policy_color if policy_color is not None else _POLICY_COLOR
    n_policies = len(policy_order)
    group_width = 0.8
    box_width = group_width / n_policies

    for ccr_idx, ccr in enumerate(_CCR_ORDER):
        for policy_idx, policy in enumerate(policy_order):
            values = data.loc[(data["CCR"] == ccr) & (data["Policy"] == policy), "TR"]
            if values.empty:
                continue
            x = ccr_idx + (policy_idx - (n_policies - 1) / 2) * box_width
            bp = ax.boxplot(
                values, positions=[x], widths=box_width * 0.9,
                patch_artist=True, showfliers=False, whis=(5, 95),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(policy_color[policy])
                patch.set_edgecolor("#0b0b0b")
                patch.set_linewidth(0.8)
            for element in ("whiskers", "caps", "medians"):
                for line in bp[element]:
                    line.set_color("#0b0b0b")
                    line.set_linewidth(1.0)

    ax.set_xticks(range(len(_CCR_ORDER)))
    ax.set_xticklabels([str(c) for c in _CCR_ORDER])
    ax.set_xlim(-0.5, len(_CCR_ORDER) - 0.5)
    ax.set_xlabel("CCR")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.axhline(1.0, color="#898781", linewidth=1, zorder=0)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=-1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=policy_color[p], edgecolor="#0b0b0b", linewidth=0.8)
        for p in policy_order
    ]
    labels = [_POLICY_LABEL_PLAIN[p] for p in policy_order]
    ax.legend(handles, labels, title="Policy", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))


# Two-series comparisons: same blue/orange pair as _BRANCH_COLOR (validated default-
# palette slots 1-2) throughout - first series is always blue, second always orange,
# regardless of which pair of configs is being compared.
_TWO_SERIES_ORDER = ["Base-Static", "T-Reschedule"]
_TWO_SERIES_COLOR = {"Base-Static": "#2a78d6", "T-Reschedule": "#eb6834"}
_TP_TWO_SERIES_ORDER = ["T-Static", "T-Reschedule"]
_TP_TWO_SERIES_COLOR = {"T-Static": "#2a78d6", "T-Reschedule": "#eb6834"}


def _draw_two_series_boxplot(
    ax, data: pd.DataFrame, title: str, y_label: str = "Throughput Ratio",
    series_order: Optional[list] = None, series_color: Optional[dict] = None,
) -> None:
    """Grouped box-and-whisker: one box per Series within each CCR group.
    `data` has columns CCR, Series, TR. Mirrors _draw_grouped_boxplot's
    style, narrowed to 2 series instead of the full policy set.

    series_order/series_color default to _TWO_SERIES_ORDER/_TWO_SERIES_COLOR
    (Base-Static vs T-Reschedule); pass overrides to plot a different pair."""
    series_order = series_order if series_order is not None else _TWO_SERIES_ORDER
    series_color = series_color if series_color is not None else _TWO_SERIES_COLOR
    n_series = len(series_order)
    group_width = 0.6
    box_width = group_width / n_series

    for ccr_idx, ccr in enumerate(_CCR_ORDER):
        for series_idx, series in enumerate(series_order):
            values = data.loc[(data["CCR"] == ccr) & (data["Series"] == series), "TR"]
            if values.empty:
                continue
            x = ccr_idx + (series_idx - (n_series - 1) / 2) * box_width
            bp = ax.boxplot(
                values, positions=[x], widths=box_width * 0.9,
                patch_artist=True, showfliers=False, whis=(5, 95),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(series_color[series])
                patch.set_edgecolor("#0b0b0b")
                patch.set_linewidth(0.8)
            for element in ("whiskers", "caps", "medians"):
                for line in bp[element]:
                    line.set_color("#0b0b0b")
                    line.set_linewidth(1.0)

    ax.set_xticks(range(len(_CCR_ORDER)))
    ax.set_xticklabels([str(c) for c in _CCR_ORDER])
    ax.set_xlim(-0.5, len(_CCR_ORDER) - 0.5)
    ax.set_xlabel("CCR")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.axhline(1.0, color="#898781", linewidth=1, zorder=0)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=-1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=series_color[s], edgecolor="#0b0b0b", linewidth=0.8)
        for s in series_order
    ]
    ax.legend(handles, series_order, title="Scheduler", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def _two_series_ratio_rows(branch: str, first_name: str, first_label: str, second_name: str, second_label: str) -> pd.DataFrame:
    """Row-level ThroughputRatio - same metric as analyze.py's heatmaps:
    Throughput / max(Throughput) among every scheduler/policy on that exact
    Workflow/CCR/Instance/Seed - restricted to the stochastic regime (where
    "reschedule" exists), for two specific full scheduler names."""
    df = _load_results(branch)
    df = df[df["Regime"] == "stochastic"].copy()
    instance_keys = ["Workflow", "CCR", "Instance", "Seed"]
    best = df.groupby(instance_keys)["Throughput"].transform("max")
    df["TR"] = df["Throughput"] / best

    rows = df[df["Scheduler"].isin([first_name, second_name])].copy()
    rows["Series"] = rows["Scheduler"].map({first_name: first_label, second_name: second_label})
    return rows


def _two_series_boxplots(
    outdir: Path, out_prefix: str, first_name_fmt: str, first_label: str,
    series_order: list, series_color: dict,
) -> list:
    """4 boxplots (HEFT/CPoP x WfCommons/RIoTBench): x-axis CCR, one box each
    for the "first" config (built from first_name_fmt.format(algo=algo)) and
    T-Reschedule, y-axis ThroughputRatio (Throughput / best-on-that-instance,
    same metric as analyze.py's heatmaps), restricted to the stochastic
    regime. Mirrors the "Ratio vs Offline" reference style (two schedulers'
    ratios against a shared best-case baseline, grouped by CCR) rather than
    TR-against-each-other."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for algo in ["HEFT", "CPoP"]:
        for branch in _BRANCH_LABEL:
            first_name = first_name_fmt.format(algo=algo)
            second_name = f"{algo}-Tp_reschedule"
            rows = _two_series_ratio_rows(branch, first_name, first_label, second_name, "T-Reschedule")
            if rows.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, 6))
            _draw_two_series_boxplot(
                ax, rows, title=f"Throughput Ratio vs Best ({algo}, {_BRANCH_LABEL[branch]})",
                series_order=series_order, series_color=series_color,
            )
            fig.tight_layout()
            name = f"{out_prefix}_{algo.lower()}_{branch}"
            out_path = outdir / f"{name}.{FIG_EXT}"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"{name} -> {out_path}")
            paths.append(out_path)
    return paths


def base_static_vs_tp_reschedule_boxplots() -> list:
    return _two_series_boxplots(
        base_static_vs_tp_reschedule_dir, "base_static_vs_tp_reschedule", "{algo}_static", "Base-Static",
        _TWO_SERIES_ORDER, _TWO_SERIES_COLOR,
    )


def tp_static_vs_tp_reschedule_boxplots() -> list:
    """Same as base_static_vs_tp_reschedule_boxplots, but comparing
    T-Static (the throughput-comparator base with no policy) instead of
    Base-Static (the plain EFT-comparator base) against T-Reschedule."""
    return _two_series_boxplots(
        tp_static_vs_tp_reschedule_dir, "tp_static_vs_tp_reschedule", "{algo}-Tp_static", "T-Static",
        _TP_TWO_SERIES_ORDER, _TP_TWO_SERIES_COLOR,
    )


def _multi_series_ratio_rows(branch: str, scheduler_labels: dict) -> pd.DataFrame:
    """Row-level ThroughputRatio - same metric as _two_series_ratio_rows /
    analyze.py's heatmaps: Throughput / max(Throughput) among every
    scheduler/policy on that exact Workflow/CCR/Instance/Seed - restricted
    to the stochastic regime, for an arbitrary {scheduler_name: label} set
    (unlike _two_series_ratio_rows, not limited to exactly 2 series)."""
    df = _load_results(branch)
    df = df[df["Regime"] == "stochastic"].copy()
    instance_keys = ["Workflow", "CCR", "Instance", "Seed"]
    best = df.groupby(instance_keys)["Throughput"].transform("max")
    df["TR"] = df["Throughput"] / best

    rows = df[df["Scheduler"].isin(scheduler_labels.keys())].copy()
    rows["Series"] = rows["Scheduler"].map(scheduler_labels)
    return rows


# 6-series comparison: reuses the validated 10-slot policy palette (blue/teal/orange
# already carry their established meaning from the 2-series plots above; green/violet
# added for the two standalone heuristics, red for MaxTP's reschedule variant).
_ALL_BASELINES_ORDER = [
    "Base-Static", "T-Static", "T-Reschedule", "FastestNode", "MaxT-Static", "MaxT-Reschedule",
]
_ALL_BASELINES_COLOR = {
    "Base-Static": "#2a78d6",       # blue
    "T-Static": "#0089a3",          # teal
    "T-Reschedule": "#eb6834",      # orange
    "FastestNode": "#4a3aa7",       # violet
    "MaxT-Static": "#008300",       # green
    "MaxT-Reschedule": "#e34948",   # red
}


def all_baselines_vs_best_boxplots() -> list:
    """Same 4 figures (HEFT/CPoP x WfCommons/RIoTBench) and same
    ThroughputRatio-vs-best-on-instance metric as
    base_static_vs_tp_reschedule_boxplots, but with 4 more reference series
    added alongside Base-Static and T-Reschedule: T-Static (the
    throughput-comparator base with no policy), the two standalone
    heuristics FastestNode and MaxT-Static (MaxTP's own bare scheduler
    name - see _maxtp_rows), and MaxT-Reschedule - the latter three are
    identical across all 4 figures since they don't depend on algo."""
    all_baselines_vs_best_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for algo in ["HEFT", "CPoP"]:
        for branch in _BRANCH_LABEL:
            scheduler_labels = {
                f"{algo}_static": "Base-Static",
                f"{algo}-Tp_static": "T-Static",
                f"{algo}-Tp_reschedule": "T-Reschedule",
                "FastestNode": "FastestNode",
                "MaxTP": "MaxT-Static",
                "MaxTP_reschedule": "MaxT-Reschedule",
            }
            rows = _multi_series_ratio_rows(branch, scheduler_labels)
            if rows.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            _draw_two_series_boxplot(
                ax, rows, title=f"Throughput Ratio vs Best ({algo}, {_BRANCH_LABEL[branch]})",
                series_order=_ALL_BASELINES_ORDER, series_color=_ALL_BASELINES_COLOR,
            )
            fig.tight_layout()
            name = f"all_baselines_vs_best_{algo.lower()}_{branch}"
            out_path = all_baselines_vs_best_dir / f"{name}.{FIG_EXT}"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"{name} -> {out_path}")
            paths.append(out_path)
    return paths


def _all_policy_ratio_rows(branch: str, scheduler_algos: list) -> pd.DataFrame:
    """Row-level ThroughputRatio - same metric as _two_series_ratio_rows /
    analyze.py's heatmaps: Throughput / max(Throughput) among every
    scheduler/policy on that exact Workflow/CCR/Instance/Seed - for every
    policy (static + all dynamic reschedule policies) of one or more
    scheduler algos on one branch (pooled together if more than one),
    stochastic regime only, pooled across every workflow.

    MaxTP's static baseline is the bare "MaxTP" scheduler name, not
    "MaxTP_static" (which is never generated - see _branch_ccr_stats)."""
    df = _load_results(branch)
    df = df[df["Regime"] == "stochastic"].copy()
    instance_keys = ["Workflow", "CCR", "Instance", "Seed"]
    best = df.groupby(instance_keys)["Throughput"].transform("max")
    df["TR"] = df["Throughput"] / best

    frames = []
    for algo in scheduler_algos:
        if algo == "MaxTP":
            static_rows = df[df["Scheduler"] == "MaxTP"].copy()
            static_rows["Policy"] = "static"
            policy_rows = df[df["Scheduler"].str.startswith("MaxTP_")].copy()
            policy_rows["Policy"] = policy_rows["Scheduler"].str.slice(len("MaxTP_"))
            algo_rows = pd.concat([static_rows, policy_rows], ignore_index=True)
        else:
            prefix = f"{algo}_"
            algo_rows = df[df["Scheduler"].str.startswith(prefix)].copy()
            algo_rows["Policy"] = algo_rows["Scheduler"].str.slice(len(prefix))
        frames.append(algo_rows)
    return pd.concat(frames, ignore_index=True)


def _all_policy_boxplot(branch: str, algos: list, title: str, out_name: str) -> Path:
    rows = _all_policy_ratio_rows(branch, algos)
    all_policy_boxplots_dir.mkdir(parents=True, exist_ok=True)
    return _save_boxplot(
        rows, title=title, y_label="Throughput Ratio",
        out_path=all_policy_boxplots_dir / f"{out_name}.{FIG_EXT}",
        policy_order=_ALL_POLICY_ORDER, policy_color=_ALL_POLICY_COLOR,
    )


def _all_policy_boxplots_with_workflows(branch: str, algos: list, title_prefix: str, out_prefix: str) -> list:
    """Aggregate boxplot (title_prefix/out_prefix, same as _all_policy_boxplot)
    plus one more per workflow."""
    rows = _all_policy_ratio_rows(branch, algos)
    all_policy_boxplots_dir.mkdir(parents=True, exist_ok=True)
    paths = [_save_boxplot(
        rows, title=f"{title_prefix} - Throughput Ratio vs Best", y_label="Throughput Ratio",
        out_path=all_policy_boxplots_dir / f"{out_prefix}.{FIG_EXT}",
        policy_order=_ALL_POLICY_ORDER, policy_color=_ALL_POLICY_COLOR,
    )]
    for workflow in sorted(rows["Workflow"].unique()):
        wf_rows = rows[rows["Workflow"] == workflow]
        paths.append(_save_boxplot(
            wf_rows, title=f"{title_prefix} - {workflow} - Throughput Ratio vs Best", y_label="Throughput Ratio",
            out_path=all_policy_boxplots_dir / f"{out_prefix}_{workflow}.{FIG_EXT}",
            policy_order=_ALL_POLICY_ORDER, policy_color=_ALL_POLICY_COLOR,
        ))
    return paths


def wfcommons_tp_heft_all_policy_boxplot() -> list:
    """WfCommons, TP-HEFT only - ThroughputRatio (vs best-on-instance, same
    metric as base_static_vs_tp_reschedule_boxplots) for every policy
    including static (no Base-Static series this time - just TP-HEFT-Static
    alongside all its dynamic reschedule policies), one box per policy per
    CCR group. Aggregate plus one per workflow."""
    return _all_policy_boxplots_with_workflows(
        "wfcommons", ["HEFT-Tp"], "WfCommons (TP-HEFT)", "wfcommons_tp_heft_all_policy_boxplot",
    )


def wfcommons_tp_cpop_all_policy_boxplot() -> list:
    """Same as wfcommons_tp_heft_all_policy_boxplot, for TP-CPoP instead of
    TP-HEFT."""
    return _all_policy_boxplots_with_workflows(
        "wfcommons", ["CPoP-Tp"], "WfCommons (TP-CPoP)", "wfcommons_tp_cpop_all_policy_boxplot",
    )


def wfcommons_tp_all_algos_all_policy_boxplot() -> Path:
    """Same as wfcommons_tp_heft_all_policy_boxplot, but pooling all three
    throughput-oriented algos together (HEFT-Tp, CPoP-Tp, MaxTP) instead of
    just one - the aggregate view of every dynamic-comparator algo, across
    every policy."""
    return _all_policy_boxplot(
        "wfcommons", ["HEFT-Tp", "CPoP-Tp", "MaxTP"],
        "WfCommons (TP-HEFT + TP-CPoP + MaxTP) - Throughput Ratio vs Best",
        "wfcommons_tp_all_algos_all_policy_boxplot",
    )


# riotbench's "train" workflow is almost all exact ties (TR == 1), which
# swamps the pooled boxplots without adding real signal - drop it there.
_EXCLUDED_WORKFLOWS = {"riotbench": {"train"}}

# Boxplots only cover the TP variant - the base (non-TP) comparators aren't
# of interest here (unlike reschedule_improvement_table/_plot, which still
# report both).
_BOXPLOT_VARIANTS = ["T"]


def _save_boxplot(
    data: pd.DataFrame, title: str, y_label: str, out_path: Path,
    policy_order: Optional[list] = None, policy_color: Optional[dict] = None,
) -> Path:
    # Wide enough for 10 policies' worth of slim boxes per CCR group plus
    # the outboard legend, without the boxes crowding together.
    fig, ax = plt.subplots(figsize=(13, 6))
    _draw_grouped_boxplot(ax, data, title=title, y_label=y_label, policy_order=policy_order, policy_color=policy_color)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"{out_path.stem} -> {out_path}")
    return out_path


def _reschedule_boxplots(metric: str, name_prefix: str, y_label: str, outdir: Path) -> list:
    """One aggregate box-and-whisker plot per Branch (TP variant only), plus
    one more per (Branch, Workflow): x-axis CCR, one box per reschedule
    policy, y-axis raw <metric> Ratio (policy/static) - same underlying
    shape as reschedule_improvement_table/_plot, faceted by branch instead
    of overlaid, and broken out by CCR instead of pooled.

    WfCommons' TP boxplots also pool in MaxTP (see _maxtp_rows) - RIoTBench's
    stay HEFT-Tp/CPoP-Tp only."""
    rows = _improvement_rows(metric)
    wfcommons_maxtp = _maxtp_rows(metric)
    wfcommons_maxtp = wfcommons_maxtp[wfcommons_maxtp["Branch"] == "wfcommons"]
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for branch in _BRANCH_LABEL:
        excluded = _EXCLUDED_WORKFLOWS.get(branch, set())
        branch_rows = rows[(rows["Branch"] == branch) & (~rows["Workflow"].isin(excluded))]
        if branch == "wfcommons":
            branch_rows = pd.concat([branch_rows, wfcommons_maxtp], ignore_index=True)
        for variant in _BOXPLOT_VARIANTS:
            subset = branch_rows[branch_rows["Variant"] == variant]
            if subset.empty:
                continue
            paths.append(_save_boxplot(
                subset, title=f"{_BRANCH_LABEL[branch]} ({variant})", y_label=y_label,
                out_path=outdir / f"{name_prefix}_{branch}_{variant.lower()}.{FIG_EXT}",
            ))
            for workflow in sorted(subset["Workflow"].unique()):
                wf_subset = subset[subset["Workflow"] == workflow]
                paths.append(_save_boxplot(
                    wf_subset, title=f"{_BRANCH_LABEL[branch]} ({variant}) - {workflow}", y_label=y_label,
                    out_path=outdir / f"{name_prefix}_{branch}_{variant.lower()}_{workflow}.{FIG_EXT}",
                ))
    return paths


def reschedule_improvement_boxplots() -> list:
    return _reschedule_boxplots("Throughput", "reschedule_boxplot", "Throughput Ratio", throughput_boxplots_dir)


def reschedule_makespan_boxplots() -> list:
    return _reschedule_boxplots("Makespan", "reschedule_makespan_boxplot", "Makespan Ratio", makespan_boxplots_dir)


def _relative_to_reschedule_rows(metric: str = "Throughput") -> pd.DataFrame:
    """Per-instance TR = <metric>_policy / <metric>_reschedule, for WfCommons's
    TP-variant algos (HEFT-Tp, CPoP-Tp, MaxTP; MaxTP's "reschedule" policy is
    a normal MaxTP_reschedule row here, unlike its missing static baseline -
    see _maxtp_rows). "reschedule" (fires after every task completion) is
    the baseline instead of static, to see how close each lighter-touch
    policy's throughput gets to always-rescheduling; "reschedule" itself is
    excluded from the policy set (trivially 1.0 against itself). Row-level
    (CCR kept, not aggregated) so callers can box-plot by CCR."""
    df = _load_results("wfcommons").copy()
    df[["Algo", "Policy"]] = df["Scheduler"].str.extract(_SCHEDULER_RE)
    df = df[df["Algo"].isin(["HEFT-Tp", "CPoP-Tp", "MaxTP"])]

    join_keys = ["Branch", "Regime", "Workflow", "CCR", "Instance", "Seed", "Algo"]
    reschedule = df.loc[df["Policy"] == "reschedule", join_keys + [metric]]
    other_policies = [p for p in _RESCHEDULE_POLICIES if p != "reschedule"]
    dynamic = df[df["Policy"].isin(other_policies)]
    merged = dynamic.merge(reschedule, on=join_keys, suffixes=("", "_reschedule"))
    merged["TR"] = merged[metric] / merged[f"{metric}_reschedule"]
    merged["Variant"] = "T"
    return merged


def reschedule_relative_boxplots() -> list:
    """Same shape as the WfCommons TP reschedule_improvement_boxplots, but
    normalized against the "reschedule" policy instead of static:
    TR = Throughput_policy / Throughput_reschedule. Shows how close each
    lighter-touch policy gets to always-rescheduling, rather than how much
    any policy beats doing nothing."""
    rows = _relative_to_reschedule_rows("Throughput")
    excluded = _EXCLUDED_WORKFLOWS.get("wfcommons", set())
    rows = rows[~rows["Workflow"].isin(excluded)]
    relative_boxplots_dir.mkdir(parents=True, exist_ok=True)
    y_label = "Throughput Ratio (vs. Reschedule)"
    paths = [_save_boxplot(
        rows, title="WfCommons (TP) - relative to Reschedule", y_label=y_label,
        out_path=relative_boxplots_dir / f"reschedule_relative_boxplot_wfcommons_tp.{FIG_EXT}",
    )]
    for workflow in sorted(rows["Workflow"].unique()):
        wf_subset = rows[rows["Workflow"] == workflow]
        paths.append(_save_boxplot(
            wf_subset, title=f"WfCommons (TP) - relative to Reschedule - {workflow}", y_label=y_label,
            out_path=relative_boxplots_dir / f"reschedule_relative_boxplot_wfcommons_tp_{workflow}.{FIG_EXT}",
        ))
    return paths


# Hand-picked (Algo, Workflow) pairs called out for discussion.
_HIGHLIGHT_CASES = [("HEFT-Tp", "blast"), ("CPoP-Tp", "montage")]


def reschedule_highlight_boxplots() -> list:
    """Throughput-improvement boxplots for two specific (Algo, Workflow)
    pairs: TP-HEFT on blast and TP-CPoP on montage (both wfcommons)."""
    rows = _improvement_rows("Throughput")
    rows = rows[rows["Branch"] == "wfcommons"]
    highlight_boxplots_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for algo, workflow in _HIGHLIGHT_CASES:
        subset = rows[(rows["Algo"] == algo) & (rows["Workflow"] == workflow)]
        if subset.empty:
            continue
        base_algo = algo.replace("-Tp", "")
        paths.append(_save_boxplot(
            subset, title=f"T-{base_algo} - {workflow.capitalize()}", y_label="Throughput Ratio",
            out_path=highlight_boxplots_dir / f"reschedule_boxplot_{base_algo.lower()}_tp_{workflow}.{FIG_EXT}",
        ))
    return paths


def _throughput_ratio_table(branch: str, table_num: int) -> Path:
    """Table of Throughput Ratio (TR) statistics for every TP-HEFT / TP-CPoP
    scheduler variant against its non-TP base scheduler, split by execution
    type (deterministic / stochastic), for a single benchmark suite. Mirrors
    the look of the Makespan Ratio table used elsewhere in the paper."""
    df = _load_results(branch)
    ratios = _throughput_ratios(df)
    agg = _aggregate(ratios)
    caption = f"Table {table_num}: Throughput Ratio statistics ({_BRANCH_LABEL[branch]})."
    tex = _render_latex(agg, caption=caption)
    name = f"throughput_ratio_table_{branch}"
    pdf_path = _compile_pdf(tex, name)
    print(f"{name} -> {pdf_path}")
    return pdf_path


def throughput_ratio_table_wfcommons() -> Path:
    return _throughput_ratio_table("wfcommons", table_num=1)


def throughput_ratio_table_riotbench() -> Path:
    return _throughput_ratio_table("riotbench", table_num=2)


FIGURES = {
    "throughput_ratio_table_wfcommons": throughput_ratio_table_wfcommons,
    "throughput_ratio_table_riotbench": throughput_ratio_table_riotbench,
    "reschedule_count_table": reschedule_count_table,
    "reschedule_improvement_table": reschedule_improvement_table,
    "wfcommons_tp_ccr_improvement_table": wfcommons_tp_ccr_improvement_table,
    "rescheduling_ccr_overview_table": rescheduling_ccr_overview_table,
    "reschedule_improvement_plot": reschedule_improvement_plot,
    "reschedule_improvement_boxplots": reschedule_improvement_boxplots,
    "reschedule_makespan_boxplots": reschedule_makespan_boxplots,
    "reschedule_highlight_boxplots": reschedule_highlight_boxplots,
    "reschedule_relative_boxplots": reschedule_relative_boxplots,
    "base_static_vs_tp_reschedule_boxplots": base_static_vs_tp_reschedule_boxplots,
    "tp_static_vs_tp_reschedule_boxplots": tp_static_vs_tp_reschedule_boxplots,
    "all_baselines_vs_best_boxplots": all_baselines_vs_best_boxplots,
    "wfcommons_tp_heft_all_policy_boxplot": wfcommons_tp_heft_all_policy_boxplot,
    "wfcommons_tp_cpop_all_policy_boxplot": wfcommons_tp_cpop_all_policy_boxplot,
    "wfcommons_tp_all_algos_all_policy_boxplot": wfcommons_tp_all_algos_all_policy_boxplot,
}


def main() -> None:
    names = sys.argv[1:] or [
        # "throughput_ratio_table_wfcommons",
        # "throughput_ratio_table_riotbench",
        # "reschedule_count_table",
        # "reschedule_improvement_table",
        # "reschedule_improvement_plot",
        # "reschedule_improvement_boxplots",
        "reschedule_highlight_boxplots",
    ]
    for name in names:
        if name not in FIGURES:
            raise SystemExit(f"Unknown figure '{name}'. Available: {list(FIGURES)}")
        FIGURES[name]()


if __name__ == "__main__":
    main()


def _faceted_two_series_boxplot(
    out_path: Path, first_name_fmt: str, first_label: str,
    series_order: list, series_color: dict, suptitle: str,
) -> Path:
    """The same four panels as _two_series_boxplots, drawn as one 2x2 figure with a
    shared legend. Rows are the algorithm (HEFT, CPoP), columns the suite
    (WfCommons, RIoTBench). Saves paper space over four separate floats."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    algos, branches = ["HEFT", "CPoP"], list(_BRANCH_LABEL)
    fig, axes = plt.subplots(len(algos), len(branches), figsize=(11, 7), sharey=True)
    for i, algo in enumerate(algos):
        for j, branch in enumerate(branches):
            ax = axes[i][j]
            rows = _two_series_ratio_rows(
                branch, first_name_fmt.format(algo=algo), first_label,
                f"{algo}-Tp_reschedule", "T-Reschedule",
            )
            if rows.empty:
                ax.set_visible(False)
                continue
            _draw_two_series_boxplot(
                ax, rows, title=f"{algo}, {_BRANCH_LABEL[branch]}",
                series_order=series_order, series_color=series_color,
            )
            if j:                       # shared y-axis: label the left column only
                ax.set_ylabel("")
            if i == 0:                  # shared x-axis: label the bottom row only
                ax.set_xlabel("")
            if ax.get_legend():
                ax.get_legend().remove()
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=series_color[s], edgecolor="black")
               for s in series_order]
    fig.legend(handles, series_order, title="Scheduler", frameon=False,
               loc="lower center", ncol=len(series_order), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"faceted -> {out_path}")
    return out_path


def tp_static_vs_tp_reschedule_faceted() -> Path:
    """Figures 5-8 of the paper as a single 2x2 float."""
    return _faceted_two_series_boxplot(
        tp_static_vs_tp_reschedule_dir / f"tp_static_vs_tp_reschedule_faceted.{FIG_EXT}",
        "{algo}-Tp_static", "T-Static", _TP_TWO_SERIES_ORDER, _TP_TWO_SERIES_COLOR,
        "Throughput Ratio vs Best: T-Static against T-Reschedule",
    )


def all_baselines_vs_best_faceted(nrows: int = 4, ncols: int = 1) -> Path:
    """The four all_baselines_vs_best panels as one float. Default 4x1 stacks them
    vertically so the figure fits a single column; pass (2, 2) for the wide form."""
    out_path = all_baselines_vs_best_dir / f"all_baselines_vs_best_faceted_{nrows}x{ncols}.{FIG_EXT}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [(algo, branch) for branch in _BRANCH_LABEL for algo in ["HEFT", "CPoP"]]
    per_w, per_h = (5.0, 2.6) if ncols == 1 else (5.5, 3.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(per_w * ncols, per_h * nrows), sharey=True)
    axes = axes.ravel()
    for ax, (algo, branch) in zip(axes, panels):
        rows = _multi_series_ratio_rows(branch, {
            f"{algo}_static": "Base-Static",
            f"{algo}-Tp_static": "T-Static",
            f"{algo}-Tp_reschedule": "T-Reschedule",
            "FastestNode": "FastestNode",
            "MaxTP": "MaxT-Static",
            "MaxTP_reschedule": "MaxT-Reschedule",
        })
        if rows.empty:
            ax.set_visible(False)
            continue
        _draw_two_series_boxplot(
            ax, rows, title=f"{algo}, {_BRANCH_LABEL[branch]}",
            series_order=_ALL_BASELINES_ORDER, series_color=_ALL_BASELINES_COLOR,
        )
        if ax is not axes[-1]:      # shared x-axis: label the bottom panel only
            ax.set_xlabel("")
        if ax.get_legend():
            ax.get_legend().remove()
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=_ALL_BASELINES_COLOR[s], edgecolor="black")
               for s in _ALL_BASELINES_ORDER]
    fig.legend(handles, _ALL_BASELINES_ORDER, title="Scheduler", frameon=False,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"faceted -> {out_path}")
    return out_path
