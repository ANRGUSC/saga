"""Scheduler-adversarial family: BIL (WINNER) beats GDL (LOSER), CORRECTED.

IMPORTANT PROVENANCE NOTE: an earlier version of this family found a ~7x
gap that turned out to be a one-line bug in saga's GDLScheduler
(`preferred_node` used `min` where the algorithm's own dynamic-level formula,
and the Dynamic Level Scheduling literature it implements, call for `max` --
higher dynamic level means a more preferred node). That bug has been fixed
directly in saga/src/saga/schedulers/gdl.py (submodule patched locally,
confirmed against upstream github.com/ANRGUSC/saga -- worth filing there).
This file is a full redo of the search against the CORRECTED scheduler.

With the bug fixed, BIL and GDL are much closer -- as expected for two
comparably sophisticated peer algorithms (both level-based list schedulers
with lookahead). The remaining, genuine gap is real but MODERATE: geomean
~1.9-1.96x at tight perturbation radii, decaying gradually (not a sharp
cliff) to ~1.3x by +-100%. It never quite clears the skill's default 2.0
STRONG threshold, but it is remarkably CONSISTENT -- p10 stays close to the
geomean itself (e.g. geomean 1.95 / p10 1.91 at frac=0.02), unlike typical
narrow comparison-fragile gaps where p10 lags far behind the geomean.

Mechanism (see HYPOTHESIS): after the fix, the two schedulers' remaining
difference is lookahead depth. BIL's per-task priority ("BIL" value) is a
fully recursive bottom-up estimate through the ENTIRE descendant subtree.
GDL's dynamic_level_2 only looks one hop ahead (largest_output_descendants:
just the single largest-output child, not further descendants). The
discovered seed is, notably, a pure task-to-node ASSIGNMENT problem (4
independent tasks, 4 heterogeneous nodes, zero dependencies) -- with no
chain to look ahead through at all, the gap here is really about how each
scheduler's OWN internal tournament/priority tie-breaking assigns a batch of
simultaneously-ready tasks to nodes of different speeds, not about deep-chain
lookahead specifically (that was the original hypothesis; the actual driver
turned out to be task-selection order for a batch of ready tasks).

Both BIL and GDL require homogeneous COMMUNICATION (all edge speeds equal)
-- see reference/saga_api.md. Do NOT use `--ccr-sweep`: `scale_to_ccr` can
rescale edges non-uniformly and would invalidate both schedulers' validity
assumptions.
"""

from __future__ import annotations

import random
from typing import Tuple

from saga import Network, TaskGraph

WINNER = "BIL"   # expected low makespan
LOSER = "GDL"    # expected high makespan

HYPOTHESIS = (
    "With saga's GDL min/max bug fixed, BIL and GDL are close peers, but a "
    "real, moderate (~1.9-1.96x), highly CONSISTENT gap remains on a pure "
    "task-to-node assignment instance (independent tasks, heterogeneous "
    "nodes, zero dependencies): BIL's per-task priority is a fully "
    "recursive bottom-up estimate through the whole descendant subtree, "
    "while GDL's default dynamic_level_2 only looks one hop ahead (the "
    "single largest-output child). For this zero-dependency seed there is "
    "no chain to look ahead through at all, so the gap instead comes from "
    "how each scheduler's own tournament/tie-break assigns several "
    "simultaneously-ready tasks to nodes of different speeds -- BIL's "
    "assignment is simply better calibrated than GDL's for this batch, and "
    "unusually consistently so (p10 stays close to the geomean, unlike a "
    "typical narrow numeric coincidence where p10 lags far behind)."
)

CLAUDE_COST_ESTIMATE = (
    "Exact, read directly from the local Claude Code session transcript "
    "(`~/.claude/projects/.../*.jsonl`) for the turns spanning this "
    "investigation -- the initial (buggy) discovery, the bug diagnosis and "
    "fix, and the full redo against the corrected scheduler. Sonnet 5, "
    "current intro pricing (through 2026-08-31): $2.00/$10.00 per 1M "
    "input/output tokens, cache write (1h TTL) $4.00/1M, cache read "
    "$0.20/1M.\n\n"
    "| | tokens | cost |\n"
    "|---|---:|---:|\n"
    "| input (uncached) | 366 | $0.00 |\n"
    "| output | 169,241 | $1.69 |\n"
    "| cache write (1h) | 815,439 | $3.26 |\n"
    "| cache read | 63,771,953 | $12.75 |\n"
    "| **total** | | **~$17.71** |"
)

# PISA-discovered seed (chain init, nodes=4), ratio 1.96, using the
# corrected GDL scheduler. Zero dependencies -- 4 independent tasks.
_BASE_NODES = {'1': 0.6981157046416699, '0': 0.4431131817225517, '2': 0.4870510411770449, '3': 0.9599097903038555}
_BASE_EDGES = {('0', '2'): 1.0, ('2', '3'): 1.0, ('1', '2'): 1.0, ('0', '1'): 1.0, ('1', '3'): 1.0, ('0', '3'): 1.0}
_BASE_TASKS = {'A': 0.5074696361445402, 'C': 0.6507183324278671, 'B': 0.4358384713035768, 'D': 0.9957993218824371}

PERTURBATION = 0.02  # tightest safe radius; geomean ~1.95, p10 ~1.91 here


def _perturb(value: float, rng: random.Random) -> float:
    lo = value * (1 - PERTURBATION)
    hi = value * (1 + PERTURBATION)
    return max(0.01, rng.uniform(lo, hi))


def make_instance(rng: random.Random) -> Tuple[Network, TaskGraph]:
    """Draw one random instance: a tight joint perturbation of the PISA seed."""
    nodes = [(name, _perturb(speed, rng)) for name, speed in _BASE_NODES.items()]
    edges = [(s, t, speed) for (s, t), speed in _BASE_EDGES.items()]  # kept =1.0 (required homogeneous comm)
    network = Network.create(nodes=nodes, edges=edges)

    tasks = [(name, _perturb(cost, rng)) for name, cost in _BASE_TASKS.items()]
    task_graph = TaskGraph.create(tasks=tasks, dependencies=[])

    return network, task_graph


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                          / ".claude/skills/find-scheduler-family/scripts"))
    from family_lib import evaluate_family, format_report

    stats = evaluate_family(make_instance, WINNER, LOSER, n=300, rng=random.Random(0))
    print(HYPOTHESIS)
    print(format_report(stats))