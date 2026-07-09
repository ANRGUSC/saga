"""Experiment: when does probability reweighting help conditional scheduling?

On generic random branching CTGs, reweighting (CHEFT) barely changes expected
makespan on average (see makespan_ratio.py). But there is a structured family,
get_problematic_instance, where reweighting reliably does much better. This
experiment shows that, and how it scales with the branch probability and the
fast/slow node speed ratio.

Metric (per instance, matching makespan_ratio.py): a scheduler's expected makespan
ratio against the clairvoyant offline schedule,

    ratio_S = sum_T p_T * (realized_makespan_S(T) / offline_makespan(T)),

where a trace's realized makespan uses the scheduler's node assignments with times
recomputed for that trace (recalculate_trace_times), and offline_makespan(T) is HEFT
scheduling just T's tasks (knowing T will run). HEFT and CHEFT are run on the SAME
instances; we report the per-instance distributions of ratio_HEFT and ratio_CHEFT
(not a single aggregate), so each scheduler is compared to offline on identical inputs.

Results (CSV + plots) are written to output/ (git-ignored).
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from saga.conditional import (
    ConditionalTaskGraph,
    recalculate_trace_times,
    schedule_trace_standalone,
)
from saga.schedulers.cheft import CheftScheduler
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    GreedyInsert,
    GreedyInsertCompareFuncs,
    UpwardRanking,
)
from saga.utils.random_graphs import get_problematic_instance

thisdir = pathlib.Path(__file__).parent.absolute()
outdir = thisdir / "output"
outdir.mkdir(exist_ok=True)

_EFT = GreedyInsertCompareFuncs.EFT
HEFT = ParametricScheduler(UpwardRanking(), GreedyInsert(compare=_EFT))
CHEFT = CheftScheduler()
OFFLINE = ParametricScheduler(UpwardRanking(), GreedyInsert(compare=_EFT))


def _realized_makespan(schedule, trace_tasks) -> float:
    task_set = set(trace_tasks)
    mapping = {
        node: [t for t in tasks if t.name in task_set]
        for node, tasks in schedule.mapping.items()
    }
    recalced = recalculate_trace_times(mapping, schedule)
    return max((t.end for tasks in recalced.values() for t in tasks), default=0.0)


def evaluate(network, ctg: ConditionalTaskGraph) -> dict:
    """Per-instance expected makespan ratio vs the clairvoyant offline, for each scheduler.

    HEFT and CHEFT are evaluated on the same instance against the same per-trace
    offline baseline.
    """
    traces = ctg.identify_traces_detailed()
    offline = [
        schedule_trace_standalone(tr["tasks"], ctg, network, OFFLINE).makespan
        for tr in traces
    ]
    s_heft = HEFT.schedule(network, ctg)
    s_cheft = CHEFT.schedule(network, ctg)
    ratio_heft = ratio_cheft = 0.0
    for tr, m_off in zip(traces, offline):
        if m_off <= 0:
            continue
        p = tr["probability"]
        ratio_heft += p * _realized_makespan(s_heft, tr["tasks"]) / m_off
        ratio_cheft += p * _realized_makespan(s_cheft, tr["tasks"]) / m_off
    return {"ratio_heft": ratio_heft, "ratio_cheft": ratio_cheft}


def run(n_instances: int, base_seed: int, **fixed) -> pd.DataFrame:
    rows = []
    for i in range(n_instances):
        np.random.seed(base_seed + i)
        network, ctg = get_problematic_instance(**fixed)
        rows.append({"seed": base_seed + i, **fixed, **evaluate(network, ctg)})
    return pd.DataFrame(rows)


_HEFT_STYLE = {"label": "HEFT", "color": "tab:red", "marker": "o", "capsize": 4}
_CHEFT_STYLE = {"label": "CHEFT", "color": "tab:blue", "marker": "s", "capsize": 4}


def main() -> None:
    # 1. Baseline family (all parameters random): per-instance ratio distributions.
    base = run(n_instances=500, base_seed=0)
    base.to_csv(outdir / "reweighting_advantage.csv", index=False)
    win_rate = (base["ratio_cheft"] < base["ratio_heft"] - 1e-9).mean()
    print("Expected makespan ratio vs clairvoyant offline, per instance (n=500):")
    for name, col in (("HEFT", "ratio_heft"), ("CHEFT", "ratio_cheft")):
        s = base[col]
        print(
            f"  {name:<5}: median={s.median():.2f}  "
            f"IQR=[{s.quantile(0.25):.2f}, {s.quantile(0.75):.2f}]  max={s.max():.2f}"
        )
    print(f"  CHEFT closer to offline than HEFT on {win_rate*100:.1f}% of instances")

    # 2. Sweep the branch probability p.
    ps = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    sweep_p = pd.concat(
        [run(n_instances=150, base_seed=1000 + int(1000 * p), likely_probability=p) for p in ps]
    )

    # 3. Sweep the fast/slow speed ratio (contention severity).
    slows = [0.8, 0.6, 0.5, 0.4, 0.3, 0.2]
    sweep_s = pd.concat(
        [run(n_instances=150, base_seed=5000 + int(1000 * s), fast_speed=2.0, slow_speed=s) for s in slows]
    )

    _plot(base, sweep_p, ps, sweep_s, slows)


def _errorbar(ax, x, df, group_col, group_vals, ratio_col, style) -> None:
    grouped = df.groupby(group_col)[ratio_col]
    mean = grouped.mean().reindex(group_vals)
    std = grouped.std().reindex(group_vals)
    ax.errorbar(x, mean.values, yerr=std.values, **style)


def _plot(base, sweep_p, ps, sweep_s, slows) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 0: per-instance ratio distributions, HEFT vs CHEFT, same instances.
    axes[0].boxplot(
        [base["ratio_heft"], base["ratio_cheft"]],
        tick_labels=["HEFT", "CHEFT"],
        showfliers=False,
    )
    axes[0].axhline(1.0, color="gray", linestyle="--", label="offline (clairvoyant)")
    axes[0].set_ylabel("expected makespan ratio vs offline")
    axes[0].set_title("Per-instance ratio (n=500)")
    axes[0].legend()

    # Panel 1: ratio vs branch probability, both schedulers, mean +/- std.
    _errorbar(axes[1], ps, sweep_p, "likely_probability", ps, "ratio_heft", _HEFT_STYLE)
    _errorbar(axes[1], ps, sweep_p, "likely_probability", ps, "ratio_cheft", _CHEFT_STYLE)
    axes[1].axhline(1.0, color="gray", linestyle="--")
    axes[1].set_xlabel("likely-branch probability p")
    axes[1].set_ylabel("expected makespan ratio vs offline")
    axes[1].set_title("Ratio vs branch probability")
    axes[1].legend()

    # Panel 2: ratio vs fast/slow speed ratio, both schedulers, mean +/- std.
    speed_ratios = [2.0 / s for s in slows]
    _errorbar(axes[2], speed_ratios, sweep_s, "slow_speed", slows, "ratio_heft", _HEFT_STYLE)
    _errorbar(axes[2], speed_ratios, sweep_s, "slow_speed", slows, "ratio_cheft", _CHEFT_STYLE)
    axes[2].axhline(1.0, color="gray", linestyle="--")
    axes[2].set_xlabel("fast / slow node speed ratio")
    axes[2].set_ylabel("expected makespan ratio vs offline")
    axes[2].set_title("Ratio vs node speed ratio")
    axes[2].legend()

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(str(outdir / "reweighting_advantage.png"))
    fig.savefig(str(outdir / "reweighting_advantage.pdf"), bbox_inches="tight")
    print(f"\nSaved plots to {outdir / 'reweighting_advantage.png'}")


if __name__ == "__main__":
    main()
