"""Heatmap showing the throughput benefit of the Throughput insertion comparator, on top of
the parametric scheduler's other knobs, for WfCommons epigenomics (deterministic regime).

X axis: (Priority function, Critical path reservation) -
    {UpwardRanking, CPoPRanking, ArbitraryTopological} x {no CP reservation, CP reservation}.
Y axis: (Insertion strategy, Append only) -
    {EFT, EST, Quickest} x {no append-only, append-only}.

Each tile is the mean, across CCR in {0.2, 0.5, 1.0, 2.0, 5.0} and problem instance, of:

    TR = Throughput[same Priority/CritPath/AppendOnly, Throughput-comparator insertion]
         / Throughput[that tile's own Priority/CritPath/Insertion/AppendOnly config]

i.e. how much throughput swapping in the Throughput comparator buys over that exact tile's
own scheduler, holding priority function, critical-path reservation, and append-only fixed.
The Throughput-comparator numerator only depends on (Priority, CritPath, AppendOnly), not on
which of EFT/EST/Quickest the tile uses, so it's computed once per (Priority, CritPath,
AppendOnly) triple and reused across all 3 insertion strategies sharing that triple.

Usage:
    python insertion_strategy_heatmap.py [n_instances]
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

from common import outputdir
from instances import base_instances, scaled
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    ArbitraryTopological, CPoPRanking, GreedyInsert, GreedyInsertCompareFuncs, UpwardRanking,
)
from saga.utils.draw import gradient_heatmap

CCRS = [0.2, 0.5, 1.0, 2.0, 5.0]
SEED = 0
WORKFLOW = "epigenomics"

_PRIORITY_FUNCS = {
    "UpwardRanking": UpwardRanking,
    "CPoPRanking": CPoPRanking,
    "ArbitraryTopological": ArbitraryTopological,
}
_PRIORITY_ORDER = list(_PRIORITY_FUNCS)
_INSERT_COMPARATORS = {
    "EFT": GreedyInsertCompareFuncs.EFT,
    "EST": GreedyInsertCompareFuncs.EST,
    "Quickest": GreedyInsertCompareFuncs.Quickest,
}
_INSERT_ORDER = list(_INSERT_COMPARATORS)


def _x_label(priority: str, critical_path: bool) -> str:
    return f"{priority} / CP={critical_path}"


def _y_label(insertion: str, append_only: bool) -> str:
    return f"{insertion} / Append={append_only}"


def _x_order(label: str):
    priority, cp = label.split(" / CP=")
    return (_PRIORITY_ORDER.index(priority), cp == "True")


def _y_order(label: str):
    insertion, append = label.split(" / Append=")
    return (_INSERT_ORDER.index(insertion), append == "True")


def _scheduler(
    priority: str, critical_path: bool, compare: GreedyInsertCompareFuncs, append_only: bool
) -> ParametricScheduler:
    return ParametricScheduler(
        initial_priority=_PRIORITY_FUNCS[priority](),
        insert_task=GreedyInsert(compare=compare, critical_path=critical_path, append_only=append_only),
    )


def _rows(n_instances: int) -> pd.DataFrame:
    base = base_instances("wfcommons", WORKFLOW, n_instances, "deterministic", seed=SEED)
    rows = []
    for ccr in CCRS:
        for instance in (scaled(b, ccr) for b in base):
            tp_throughput = {}
            for priority in _PRIORITY_FUNCS:
                for critical_path in (False, True):
                    for append_only in (False, True):
                        scheduler = _scheduler(
                            priority, critical_path, GreedyInsertCompareFuncs.Throughput, append_only
                        )
                        schedule = scheduler.schedule(instance.network, instance.task_graph)
                        tp_throughput[(priority, critical_path, append_only)] = schedule.throughput

            for priority in _PRIORITY_FUNCS:
                for critical_path in (False, True):
                    for insertion, compare in _INSERT_COMPARATORS.items():
                        for append_only in (False, True):
                            scheduler = _scheduler(priority, critical_path, compare, append_only)
                            schedule = scheduler.schedule(instance.network, instance.task_graph)
                            tp = tp_throughput[(priority, critical_path, append_only)]
                            rows.append({
                                "X": _x_label(priority, critical_path),
                                "Y": _y_label(insertion, append_only),
                                "TR": tp / schedule.throughput,
                            })
    return pd.DataFrame(rows)


def main() -> None:
    n_instances = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    df = _rows(n_instances)

    outdir = outputdir / "wfcommons_deterministic"
    outdir.mkdir(parents=True, exist_ok=True)
    ax = gradient_heatmap(
        df, x="X", y="Y", color="TR",
        title=f"wfcommons / deterministic: {WORKFLOW} (Throughput-comparator benefit)",
        x_label="Priority function / Critical path reservation",
        y_label="Insertion strategy / Append only",
        xorder=_x_order, yorder=_y_order,
        cmap="coolwarm_r",  # high (TP comparator much better) is cool, low is warm/red
        upper_threshold=2.0,  # clip the gradient/colorbar at 2.0; anything above saturates dark blue
        color_center=1.0,  # white at TR == 1.0 (no benefit); red only below that
        clip_cell_text=False,  # cell labels always show the actual mean, not ">2.0"
        cell_font_size=12, font_size=12, figsize=(11, 9),
    )
    out_path = outdir / f"{WORKFLOW}_insertion_strategy_heatmap.png"
    ax.get_figure().savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(ax.get_figure())
    print(f"insertion_strategy_heatmap -> {out_path}")


if __name__ == "__main__":
    main()
