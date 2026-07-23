"""Demonstrates SAGA's support for task duplication.

SAGA's Schedule can place a task on more than one node, so a successor can read a
local copy instead of paying the transfer cost. HEFT exposes this via
``duplication_factor``. The heuristic here is deliberately naive (duplicate every
communication-heavy task onto its best nodes); the point is that the framework
supports duplication, not that this particular strategy is good.

This sweeps CCR using Network.scale_to_ccr (which preserves link heterogeneity)
and compares HEFT across duplication factors, averaged over random branching DAGs.
The naive strategy often over-duplicates at high CCR, so higher factors can hurt.
"""

import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np

from saga.schedulers.heft import HeftScheduler
from saga.utils.random_graphs import get_branching_dag, get_network

thisdir = pathlib.Path(__file__).resolve().parent

CCRS = [0.5, 1.0, 2.0, 5.0, 10.0]
DUPLICATION_FACTORS = [1, 2, 3]
NUM_INSTANCES = 20


def main() -> None:
    random.seed(0)
    np.random.seed(0)

    # makespans[factor][ccr] = list of makespans over instances
    makespans = {f: {ccr: [] for ccr in CCRS} for f in DUPLICATION_FACTORS}
    for _ in range(NUM_INSTANCES):
        task_graph = get_branching_dag(levels=3, branching_factor=2)
        base_network = get_network(num_nodes=4)
        for ccr in CCRS:
            network = base_network.scale_to_ccr(task_graph, ccr)
            for factor in DUPLICATION_FACTORS:
                schedule = HeftScheduler(duplication_factor=factor).schedule(
                    network, task_graph
                )
                makespans[factor][ccr].append(schedule.makespan)

    print(f"{'CCR':>6}  " + "  ".join(f"factor={f:>2}" for f in DUPLICATION_FACTORS))
    for ccr in CCRS:
        means = [float(np.mean(makespans[f][ccr])) for f in DUPLICATION_FACTORS]
        print(f"{ccr:>6}  " + "  ".join(f"{m:>9.3f}" for m in means))

    # Plot mean makespan vs CCR, one line per duplication factor.
    plt.figure(figsize=(7, 4))
    for factor in DUPLICATION_FACTORS:
        means = [float(np.mean(makespans[factor][ccr])) for ccr in CCRS]
        plt.plot(CCRS, means, marker="o", label=f"duplication_factor={factor}")
    plt.xlabel("CCR (communication / computation)")
    plt.ylabel("mean makespan")
    plt.title("HEFT with task duplication vs CCR")
    plt.legend()
    plt.tight_layout()
    out = thisdir / "duplication_vs_ccr.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
