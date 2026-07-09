"""Experiment: conditional schedulers vs. the clairvoyant offline schedule.

A conditional scheduler commits ONE overlapping schedule over all branches of a
conditional task graph, fixing each task's node offline. At runtime a single
trace executes; its tasks run on their assigned nodes as early as their in-trace
dependencies allow. We measure how much worse that committed schedule is than a
clairvoyant scheduler that knew which trace would run.

For each trace T with probability p_T:
    r_S(T) = realized_makespan_S(T) / offline_makespan(T)
      realized_makespan_S(T): tasks of T under scheduler S's schedule, node
        assignments fixed, start times recomputed for T alone (recalculate_trace_times).
      offline_makespan(T): HEFT scheduling only T's tasks (a plain DAG).
Instance metric: expected ratio E[r_S] = sum_T p_T r_S(T)  (>= ~1).

Compared approaches (all produce overlapping conditional schedules; one shared
clairvoyant offline denominator):
    HEFT   - UpwardRanking + EFT insert           (exploits overlap, no reweight)
    CHEFT  - ProbabilityWeighted(UpwardRanking)   (reweights priority by branch prob)
    CPoP   - CPoPRanking + EFT insert + critical-path pin
    CCPoP  - ProbabilityWeighted(CPoPRanking)     (reweighted CPoP)

Random conditional branching DAGs; results written to output/ (git-ignored).
"""

import itertools
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from saga import Network, Schedule, TaskGraph
from saga.conditional import (
    ConditionalTaskGraph,
    recalculate_trace_times,
    schedule_trace_standalone,
)
from saga.overlap import NoOverlapPolicy
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    CPoPRanking,
    GreedyInsert,
    GreedyInsertCompareFuncs,
    ProbabilityWeighted,
    UpwardRanking,
)
from saga.utils.random_graphs import get_network, get_random_conditional_branching_dag

thisdir = pathlib.Path(__file__).parent.absolute()
outdir = thisdir / "output"
outdir.mkdir(exist_ok=True)

_EFT = GreedyInsertCompareFuncs.EFT


class _NoOverlapScheduler(ParametricScheduler):
    """Overlap-oblivious baseline: schedules the CTG ignoring mutual exclusion.

    Forces a NoOverlapPolicy so mutually exclusive branches are serialized like an
    ordinary DAG. Shows the value of exploiting overlap at all.
    """

    def schedule(self, network, task_graph, schedule=None, min_start_time=0.0,
                 node_constraints=None) -> Schedule:
        if schedule is None:
            schedule = Schedule(task_graph, network, overlap_policy=NoOverlapPolicy())
        return super().schedule(network, task_graph, schedule=schedule,
                                min_start_time=min_start_time)


# All conditional approaches share the greedy EFT insert; they differ only in the
# priority (and, for CPoP, the critical-path pin). Reweighting = wrap the priority
# in ProbabilityWeighted. "HEFT-no-overlap" is the overlap-oblivious baseline.
CONDITIONAL_SCHEDULERS = {
    "HEFT-no-overlap": _NoOverlapScheduler(UpwardRanking(), GreedyInsert(compare=_EFT)),
    "HEFT": ParametricScheduler(UpwardRanking(), GreedyInsert(compare=_EFT)),
    "CHEFT": ParametricScheduler(
        ProbabilityWeighted(base=UpwardRanking()), GreedyInsert(compare=_EFT)
    ),
    "CPoP": ParametricScheduler(
        CPoPRanking(), GreedyInsert(compare=_EFT, critical_path=True)
    ),
    "CCPoP": ParametricScheduler(
        ProbabilityWeighted(base=CPoPRanking()),
        GreedyInsert(compare=_EFT, critical_path=True),
    ),
}
# Clairvoyant per-trace baseline (knows the trace; schedules it as a plain DAG).
OFFLINE_SCHEDULER = ParametricScheduler(UpwardRanking(), GreedyInsert(compare=_EFT))


def _makespan(mapping) -> float:
    return max((t.end for tasks in mapping.values() for t in tasks), default=0.0)


def realized_makespan(schedule: Schedule, trace_tasks) -> float:
    """Makespan of one trace under a committed schedule, times recomputed for that trace.

    Node assignments come from the conditional schedule; start times are the
    earliest feasible once only this trace's tasks run.
    """
    task_set = set(trace_tasks)
    trace_mapping = {
        node: [t for t in tasks if t.name in task_set]
        for node, tasks in schedule.mapping.items()
    }
    return _makespan(recalculate_trace_times(trace_mapping, schedule))


def evaluate_instance(network, ctg: ConditionalTaskGraph) -> tuple[int, dict]:
    """Return (n_traces, {algo: {metric: value}}) for one CTG/network instance."""
    traces = ctg.identify_traces_detailed()
    offline = [
        schedule_trace_standalone(tr["tasks"], ctg, network, OFFLINE_SCHEDULER).makespan
        for tr in traces
    ]

    results: dict = {}
    for algo, scheduler in CONDITIONAL_SCHEDULERS.items():
        schedule = scheduler.schedule(network, ctg)
        exp_ratio = exp_realized = exp_offline = 0.0
        worst_ratio = 0.0
        for i, trace in enumerate(traces):
            p = trace["probability"]
            m_s = realized_makespan(schedule, trace["tasks"])
            m_off = offline[i]
            ratio = m_s / m_off if m_off > 0 else 1.0
            exp_ratio += p * ratio
            exp_realized += p * m_s
            exp_offline += p * m_off
            worst_ratio = max(worst_ratio, ratio)
        results[algo] = {
            "exp_ratio": exp_ratio,
            "ratio_of_exp": exp_realized / exp_offline if exp_offline > 0 else 1.0,
            "worst_ratio": worst_ratio,
        }
    return len(traces), results


def run(configs, n_instances: int, base_seed: int = 0) -> pd.DataFrame:
    rows = []
    total = len(configs) * n_instances
    done = 0
    for cfg in configs:
        for i in range(n_instances):
            seed = base_seed + done
            np.random.seed(seed)
            ctg = get_random_conditional_branching_dag(
                levels=cfg["levels"],
                branching_factor=cfg["branching_factor"],
                conditional_parent_probability=cfg["cond_prob"],
            )
            network = get_network(cfg["num_nodes"])
            n_traces, results = evaluate_instance(network, ctg)
            for algo, metrics in results.items():
                rows.append(
                    {**cfg, "seed": seed, "scheduler": algo, "n_traces": n_traces, **metrics}
                )
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  {done}/{total} instances", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    conditional = df[df["n_traces"] > 1]  # only instances with real branching
    print(
        f"\n{len(conditional)//len(CONDITIONAL_SCHEDULERS)} conditional instances "
        f"(>1 trace) of {len(df)//len(CONDITIONAL_SCHEDULERS)} total\n"
    )
    print("Expected makespan ratio vs clairvoyant offline (conditional instances only):")
    summary = (
        conditional.groupby("scheduler")["exp_ratio"]
        .agg(mean="mean", median="median", p90=lambda s: s.quantile(0.90), max="max")
        .reindex(list(CONDITIONAL_SCHEDULERS))
    )
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nBy #network nodes (mean exp_ratio):")
    pivot = conditional.pivot_table(
        index="num_nodes", columns="scheduler", values="exp_ratio", aggfunc="mean"
    )[list(CONDITIONAL_SCHEDULERS)]
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nReweighting effect (mean exp_ratio, lower is better):")
    for base, reweighted in (("HEFT", "CHEFT"), ("CPoP", "CCPoP")):
        b = conditional[conditional["scheduler"] == base]["exp_ratio"].mean()
        r = conditional[conditional["scheduler"] == reweighted]["exp_ratio"].mean()
        print(f"  {base} {b:.4f} -> {reweighted} {r:.4f}  ({100*(b-r)/b:+.1f}%)")


def plot(df: pd.DataFrame, path: pathlib.Path) -> None:
    conditional = df[df["n_traces"] > 1]
    order = list(CONDITIONAL_SCHEDULERS)
    data = [conditional[conditional["scheduler"] == a]["exp_ratio"].values for a in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="offline (clairvoyant)")
    ax.set_ylabel("Expected makespan ratio vs offline")
    ax.set_title("Conditional scheduling vs clairvoyant offline")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(str(path))
    print(f"\nSaved boxplot to {path}")


def main() -> None:
    grid = {
        "num_nodes": [2, 4, 8],
        "levels": [3, 4],
        "branching_factor": [2],
        "cond_prob": [0.4, 0.8],
    }
    configs = [
        dict(zip(grid, values)) for values in itertools.product(*grid.values())
    ]
    n_instances = 40
    print(f"{len(configs)} configs x {n_instances} instances")
    df = run(configs, n_instances=n_instances)
    df.to_csv(outdir / "makespan_ratio.csv", index=False)
    summarize(df)
    plot(df, outdir / "makespan_ratio_boxplot.png")


if __name__ == "__main__":
    main()
