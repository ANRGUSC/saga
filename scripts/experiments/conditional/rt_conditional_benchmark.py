"""Benchmark: conditional schedulers on the real-time-grounded conditional-DAG family.

Uses get_conditional_dag, a nested series/parallel/conditional generator following the
real-time-systems conditional DAG task model (Melani et al., ECRTS 2015) with per-branch
probabilities (Ueter, Guenzel, Chen, 2021). This is the principled synthetic benchmark
(route 1 from the literature review), replacing the ad-hoc get_random_conditional_branching_dag.

For each instance we report every scheduler's per-instance expected makespan ratio vs the
clairvoyant offline (see _eval.py). We also sweep the branch-probability skew (Dirichlet
alpha): skewed branches (small alpha) are where probability reweighting should help most.

Results (CSV + plot) written to output/ (git-ignored).
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _eval import SCHEDULERS, expected_makespan_ratios
from saga.utils.random_graphs import get_conditional_dag, get_network

thisdir = pathlib.Path(__file__).parent.absolute()
outdir = thisdir / "output"
outdir.mkdir(exist_ok=True)

ALGOS = list(SCHEDULERS)


def run(n_instances: int, base_seed: int, num_nodes=None, **gen_kwargs) -> pd.DataFrame:
    rows = []
    for i in range(n_instances):
        np.random.seed(base_seed + i)
        ctg = get_conditional_dag(**gen_kwargs)
        nodes = num_nodes if num_nodes is not None else int(np.random.randint(2, 9))
        network = get_network(nodes)
        n_traces = len(ctg.identify_traces_detailed())
        rows.append(
            {
                "seed": base_seed + i,
                "num_nodes": nodes,
                "n_traces": n_traces,
                **gen_kwargs,
                **expected_makespan_ratios(network, ctg),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    # 1. Corpus: mixed sizes/nodes, moderately skewed conditional branches.
    corpus = run(
        n_instances=400,
        base_seed=0,
        max_depth=3,
        branch_factor=3,
        conditional_probability=0.6,
        branch_alpha=0.7,
    )
    corpus.to_csv(outdir / "rt_conditional_benchmark.csv", index=False)
    conditional = corpus[corpus["n_traces"] > 1]
    print(
        f"Corpus: {len(conditional)}/{len(corpus)} instances have >1 trace "
        f"(median {int(conditional['n_traces'].median())} traces)\n"
    )
    print("Expected makespan ratio vs clairvoyant offline (conditional instances):")
    for algo in ALGOS:
        s = conditional[algo]
        print(
            f"  {algo:<6}: median={s.median():.3f}  "
            f"IQR=[{s.quantile(0.25):.3f}, {s.quantile(0.75):.3f}]  mean={s.mean():.3f}"
        )
    for base, rw in (("HEFT", "CHEFT"), ("CPoP", "CCPoP")):
        better = (conditional[rw] < conditional[base] - 1e-9).mean()
        print(f"  {rw} better than {base} on {better*100:.1f}% of instances")

    # 2. Sweep branch-probability skew (Dirichlet alpha): small = skewed branches.
    alphas = [0.2, 0.5, 1.0, 2.0, 5.0]
    sweep = pd.concat(
        [
            run(
                n_instances=150,
                base_seed=10000 + int(1000 * a),
                num_nodes=4,
                max_depth=3,
                branch_factor=3,
                conditional_probability=0.7,
                branch_alpha=a,
            )
            for a in alphas
        ]
    )

    _plot(conditional, sweep, alphas)


def _plot(corpus, sweep, alphas) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].boxplot([corpus[a] for a in ALGOS], tick_labels=ALGOS, showfliers=False)
    axes[0].axhline(1.0, color="gray", linestyle="--", label="offline (clairvoyant)")
    axes[0].set_ylabel("expected makespan ratio vs offline")
    axes[0].set_title("RT-grounded conditional DAGs (per instance)")
    axes[0].legend()

    for algo in ALGOS:
        grouped = sweep.groupby("branch_alpha")[algo]
        mean = grouped.mean().reindex(alphas)
        std = grouped.std().reindex(alphas)
        axes[1].errorbar(alphas, mean.values, yerr=std.values, marker="o", capsize=4, label=algo)
    axes[1].axhline(1.0, color="gray", linestyle="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("branch skew: Dirichlet alpha (small = skewed)")
    axes[1].set_ylabel("expected makespan ratio vs offline")
    axes[1].set_title("Ratio vs branch-probability skew")
    axes[1].legend()

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(str(outdir / "rt_conditional_benchmark.png"))
    fig.savefig(str(outdir / "rt_conditional_benchmark.pdf"), bbox_inches="tight")
    print(f"\nSaved plot to {outdir / 'rt_conditional_benchmark.png'}")


if __name__ == "__main__":
    main()
