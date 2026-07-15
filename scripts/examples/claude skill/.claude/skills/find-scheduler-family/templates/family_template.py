"""Template for a scheduler-adversarial problem-instance FAMILY.

A family is a distribution over (Network, TaskGraph) instances, expressed as
a function `make_instance(rng)`. The goal: instances drawn from it should
make WINNER's expected makespan much lower than LOSER's.

START HERE: perturb the seed, don't freely re-randomize it.
--------------------------------------------------------------------------
seed_pisa.py's `*_summary.json` reports each parameter's MARGINAL range
across the one discovered instance (e.g. "node speeds span 0.10-0.61"). That
is NOT a safe sampling distribution -- independently redrawing each
parameter across its own marginal range usually destroys the specific JOINT
alignment (which rank/EFT comparison wins, which task contends with which)
that made the seed adversarial, even though every individual draw is
"in range". This silently collapses a real gap to no-effect and is the most
common way this template gets misused.

The correct first move: run `family_lib.sweep_perturbation` on the exact
seed instance to find how wide a joint perturbation radius survives, THEN
build `make_instance` as a perturbation of the seed at a radius just inside
that cliff (see `family_lib.perturb_instance`). Only attempt a broader,
freely-restructured family if the sweep shows the effect survives out past
~50-100% -- that indicates a CATEGORICAL mechanism (robust to almost any
numbers), not a COMPARISON-FRAGILE one (only real in a narrow neighborhood).

    python ../scripts/seed_pisa.py --winner HEFT --loser FastestNode ...
    # then, in a scratch script:
    import json, random
    from saga import Network, TaskGraph
    from family_lib import sweep_perturbation, format_perturbation_sweep

    net = Network(**json.load(open("seeds/HEFT_vs_FastestNode_network.json")))
    tg = TaskGraph(**json.load(open("seeds/HEFT_vs_FastestNode_taskgraph.json")))
    results = sweep_perturbation(net, tg, "HEFT", "FastestNode", random.Random(0))
    print(format_perturbation_sweep(results))
    # read off the largest frac where geomean>=threshold AND p10>=1.2;
    # use something comfortably inside that (e.g. half the cliff radius).

Then validate the chosen family with:

    python ../scripts/benchmark_family.py --family ./this_file.py --samples 300

Keep it a genuine DISTRIBUTION (a perturbation ball is one, since it's
uncountably many distinct instances -- not a single hard-coded instance).
"""

from __future__ import annotations

import random
from typing import Tuple

from saga import Network, TaskGraph

# Declare the pairing so benchmark_family.py can pick it up without CLI flags.
WINNER = "HEFT"          # expected low makespan
LOSER = "FastestNode"    # expected high makespan

# Human-readable statement of the mechanism this family exploits. Update it.
HYPOTHESIS = (
    "With a wide, parallel DAG on a heterogeneous network, FastestNode piles "
    "every task onto the single fastest node and serializes them, while HEFT "
    "spreads independent tasks across nodes to run them in parallel."
)

# --- Option A: categorical mechanism -> broad structural family -----------
# Use this style if sweep_perturbation showed the gap survives out past
# ~50-100% perturbation (robust to almost any numbers, e.g. FastestNode
# ignoring parallelism entirely). Vary topology, counts, and weights freely.


def make_instance(rng: random.Random) -> Tuple[Network, TaskGraph]:
    """Draw one random instance from the family.

    MUST accept a random.Random and return (Network, TaskGraph). Use `rng` for
    ALL randomness so results are reproducible under a fixed seed.
    """
    # --- network: heterogeneous compute, fast communication -------------------
    num_nodes = rng.randint(3, 6)
    nodes = [(str(i), rng.uniform(0.1, 1.0)) for i in range(num_nodes)]
    # Fully-connected fast links so parallelism, not comm, dominates.
    # Network is UNDIRECTED: add each pair once (i < j). Self-loops default to inf.
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edges.append((str(i), str(j), rng.uniform(5.0, 10.0)))
    network = Network.create(nodes=nodes, edges=edges)

    # --- task graph: many independent parallel tasks --------------------------
    width = rng.randint(4, 8)
    tasks = [("root", 0.01)]
    deps = []
    for k in range(width):
        name = f"t{k}"
        tasks.append((name, rng.uniform(0.5, 1.0)))
        deps.append(("root", name, 0.01))   # tiny data => cheap to place anywhere
    task_graph = TaskGraph.create(tasks=tasks, dependencies=deps)

    return network, task_graph


# --- Option B: comparison-fragile mechanism -> perturb the exact seed -----
# Use this style if sweep_perturbation showed the gap only survives to
# ~10-20% (e.g. HEFT vs CPoP: the gap depends on a rank/EFT tie-break, not a
# categorical rule). Hardcode the seed's exact values and perturb jointly.
#
# import pathlib, sys
# sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
# from family_lib import perturb_instance
#
# _BASE_NODES = {...}   # copy exact values from *_network.json
# _BASE_EDGES = {...}
# _BASE_TASKS = {...}   # copy exact values from *_taskgraph.json
# _BASE_DEPS = {...}
# PERTURBATION = 0.10   # set from the sweep: comfortably inside the cliff
#
# def make_instance(rng: random.Random) -> Tuple[Network, TaskGraph]:
#     nodes = [(name, ...) for name, speed in _BASE_NODES.items()]
#     ...
#     network = Network.create(nodes=nodes, edges=edges)
#     task_graph = TaskGraph.create(tasks=tasks, dependencies=deps)
#     return network, task_graph


if __name__ == "__main__":
    # Quick self-check.
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from family_lib import evaluate_family, format_report

    stats = evaluate_family(make_instance, WINNER, LOSER, n=200, rng=random.Random(0))
    print(HYPOTHESIS)
    print(format_report(stats))