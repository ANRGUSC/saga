"""Scheduler-adversarial family: HEFT (winner) vs CPoP (loser).

Provenance: discovered via PISA simulated annealing on a 5-node / 8-task
branching DAG (seed_pisa.py --winner HEFT --loser CPoP --init branching
--nodes 5 --seed 7 --restarts 16 --iterations 1200), which found an instance
with makespan(CPoP)/makespan(HEFT) = 2.322.

`family_lib.sweep_perturbation` on that seed showed the gap is
COMPARISON-FRAGILE (survives only to a ~15-20% joint perturbation radius,
not the ~50-100% that would indicate a categorical mechanism):

    frac   geomean  median   p10    frac>=2.0x
    0.02    2.281    2.313  2.262     93.2%
    0.05    2.118    2.242  1.810     65.0%
    0.07    2.064    2.191  1.760     59.4%
    0.10    1.864    1.845  1.603     38.4%
    0.15    1.687    1.732  1.181     22.0%   <- p10 starts dropping below 1.2
    0.20    1.510    1.611  1.000     14.6%   <- median starts collapsing to ties

So this family IS the seed instance's exact topology and weights, jointly
perturbed by +-7% (PERTURBATION below) -- comfortably inside the cliff at
~15%, with real headroom to spare (STRONG at this radius: geomean 2.06,
p10 1.76).
"""

from __future__ import annotations

import random
from typing import Tuple

from saga import Network, TaskGraph

WINNER = "HEFT"
LOSER = "CPoP"

HYPOTHESIS = (
    "CPoP ranks tasks by (upward rank + downward rank) and forces every task "
    "on the critical path onto a single fixed 'critical path processor' "
    "(the node minimizing total CP execution time), scheduling everything "
    "else around that commitment. In this DAG the critical path threads "
    "through several tasks that would be better placed on different nodes "
    "once contention is considered, so CPoP's single-processor commitment "
    "creates a serialization bottleneck. HEFT has no such constraint -- it "
    "picks the earliest-finish-time node independently for every task -- so "
    "it routes around the contention CPoP creates for itself. The effect "
    "depends on the exact rank tie-break / EFT comparisons in this instance "
    "(confirmed comparison-fragile by the perturbation sweep above), so the "
    "family stays close to the discovered seed rather than re-randomizing "
    "topology freely."
)

# Radius chosen from the sweep above: comfortably inside the ~15-20% cliff
# where p10 collapses, while still STRONG (geomean >= 2.0, p10 >= 1.2).
PERTURBATION = 0.07

# --- exact seed values (outputs/HEFT_vs_CPoP/seeds/HEFT_vs_CPoP_*.json) ---

_BASE_NODES = {
    "1": 0.39568209321205655,
    "4": 0.40417370737290914,
    "0": 1.0,
    "2": 0.2270666210411646,
    "3": 0.4670001534339512,
}

# Undirected network: each pair listed once (source, target, speed).
# Self-loops are omitted -- SAGA defaults intra-node comm to free (inf).
_BASE_EDGES = [
    ("2", "3", 0.46634984533153867),
    ("2", "4", 0.42109110943798156),
    ("0", "1", 0.7737527062110012),
    ("0", "3", 0.23036910456179632),
    ("0", "4", 1.0),
    ("1", "3", 0.11669757426443791),
    ("1", "4", 0.6411661120140184),
    ("0", "2", 0.31177543698756516),
    ("3", "4", 0.5136169104925971),
    ("1", "2", 0.45250456762429647),
]

_BASE_TASKS = {
    "1": 0.2777000009758267,
    "4": 0.7818838688460732,
    "0": 0.4469302381511251,
    "2": 0.19749295826410102,
    "3": 0.3095046026564685,
    "7": 0.3838168813683075,
    "6": 0.3507858408721879,
    "5": 1.0,
}

_BASE_DEPS = [
    ("6", "1", 0.3898345142933203),
    ("7", "0", 0.881037552588612),
    ("6", "3", 0.29812235566867795),
    ("2", "4", 0.32911981951190555),
    ("3", "5", 0.4446985960493052),
    ("2", "6", 0.47275820266304297),
    ("4", "5", 0.7308589653490813),
    ("7", "1", 0.6390166622520582),
    ("0", "1", 0.7837910598476928),
    ("7", "4", 0.31517814131138433),
    ("6", "0", 0.4774720613051818),
    ("1", "4", 0.3357167623301755),
    ("7", "3", 0.2658790115006339),
    ("7", "5", 0.8675867111643873),
    ("0", "5", 0.3928108771439245),
    ("1", "5", 0.5002531023175185),
    ("2", "1", 0.6921066657791345),
]


def make_instance(rng: random.Random) -> Tuple[Network, TaskGraph]:
    """Draw one instance: the PISA seed, jointly perturbed by +-PERTURBATION.

    Keeps the exact topology (same nodes, tasks, dependency structure) of
    the adversarial seed and only resamples each weight within
    value * (1 +- PERTURBATION), preserving the rank/EFT comparisons that
    make CPoP's critical-path commitment backfire.
    """
    def bump(v: float) -> float:
        return max(1e-6, rng.uniform(v * (1 - PERTURBATION), v * (1 + PERTURBATION)))

    nodes = [(name, bump(speed)) for name, speed in _BASE_NODES.items()]
    edges = [(s, t, bump(speed)) for s, t, speed in _BASE_EDGES]
    network = Network.create(nodes=nodes, edges=edges)

    tasks = [(name, bump(cost)) for name, cost in _BASE_TASKS.items()]
    deps = [(s, t, bump(size)) for s, t, size in _BASE_DEPS]
    task_graph = TaskGraph.create(tasks=tasks, dependencies=deps)

    return network, task_graph


if __name__ == "__main__":
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / ".claude/skills/find-scheduler-family/scripts"))
    from family_lib import evaluate_family, format_report

    stats = evaluate_family(make_instance, WINNER, LOSER, n=200, rng=random.Random(0))
    print(HYPOTHESIS)
    print(format_report(stats))
