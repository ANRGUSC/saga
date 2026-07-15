"""Scheduler-adversarial family (WEAK/INCONSISTENT): CPoP vs ETF.

WINNER/LOSER here is nominal -- see HYPOTHESIS. Extensive search (PISA in
both directions, perturbation sweeps on 3 independent seeds) found no STRONG
family in either direction. This file keeps the best available lead: ETF
mildly, but inconsistently, outperforms CPoP.

Mechanism, confirmed by reading source (saga/schedulers/cpop.py,
saga/schedulers/etf.py) and the Network model (saga/__init__.py):

ETF requires homogeneous compute (all node speeds equal -- see
reference/saga_api.md). CPoP's critical-path-processor rule picks
`argmin(sum(critical_task.cost) / node.speed)` over all nodes. When every
node has the SAME speed (forced by ETF's constraint), this expression is
IDENTICAL for every node -- a flat tie. Python's `min()` then just returns
whichever node happens to come first in `Network.nodes`'s iteration order,
which is a `FrozenSet[NetworkNode]` (see saga/__init__.py) -- i.e. hash-order
dependent and uncorrelated with the instance's actual communication
topology. So under this required constraint, CPoP ends up pinning its
critical path to an effectively ARBITRARY node, not a deliberately-chosen
one. Sometimes that arbitrary node is well-connected (CPoP does fine or
even better, since it still avoids communication among pinned tasks);
sometimes it's poorly connected (CPoP does badly, and ETF -- which isn't
pinned to anything, just less thorough about idle-gap insertion -- wins).

This produces a real but NOISY, uncontrollable lean: perturbation sweeps on
three independent PISA seeds (both directions, branching and chain init)
consistently gave geomean ~1.0-1.3 favoring ETF with p10 BELOW 1.0 (i.e.
CPoP still wins a meaningful fraction of the time) -- never STRONG, never
zero. A first attempt to push the direction that favors CPoP additionally
turned out to be a near-zero-value artifact: allowing perturbed weights to
approach 0 inflated the apparent ratio (geomean up to 2.7 at frac=1.0) via
degenerate near-instant tasks/communication, not a real structural effect --
this collapsed back to ~1.0-1.1 once a sane floor was added (values can't
fall below 30% of their base). See conversation history / SKILL.md's
perturbation-sweep guidance for how this was diagnosed.

Reported honestly as WEAK/INCONSISTENT, not a successful STRONG family.
"""

from __future__ import annotations

import pathlib
import random
import sys
from typing import Tuple

from saga import Network, TaskGraph

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                      / ".claude/skills/find-scheduler-family/scripts"))
from family_lib import perturb_instance  # noqa: E402

WINNER = "ETF"    # mild, inconsistent lean -- see HYPOTHESIS
LOSER = "CPoP"

HYPOTHESIS = (
    "ETF requires homogeneous compute (all node speeds equal). Under that "
    "constraint, CPoP's critical-path-processor rule -- argmin(sum(critical "
    "task costs) / node.speed) -- becomes a flat tie across every node (all "
    "speeds equal), so Python's min() just returns whichever node happens to "
    "come first in Network.nodes's frozenset iteration order: effectively "
    "arbitrary, uncorrelated with the instance's real communication "
    "topology. So CPoP pins its critical path to a node chosen for no good "
    "reason -- sometimes that's fine, sometimes it's badly connected and "
    "hurts CPoP, while ETF (unpinned, but unable to insert into idle "
    "schedule gaps) is not affected by this specific failure mode. This "
    "gives a real but NOISY, largely uncontrollable lean toward ETF "
    "(observed geomean ~1.2-1.3 across sweeps) with p10 consistently BELOW "
    "1.0 -- CPoP still wins a meaningful fraction of instances, because "
    "which node gets arbitrarily pinned is essentially a coin flip from the "
    "family generator's point of view."
)

CLAUDE_COST_ESTIMATE = (
    "Exact, read directly from the local Claude Code session transcript "
    "(`~/.claude/projects/.../*.jsonl`) for the turns spanning this "
    "investigation. Sonnet 5, current intro pricing (through 2026-08-31): "
    "$2.00/$10.00 per 1M input/output tokens, cache write (1h TTL) "
    "$4.00/1M, cache read $0.20/1M.\n\n"
    "| | tokens | cost |\n"
    "|---|---:|---:|\n"
    "| input (uncached) | 136 | $0.00 |\n"
    "| output | 81,761 | $0.82 |\n"
    "| cache write (1h) | 106,533 | $0.43 |\n"
    "| cache read | 18,134,153 | $3.63 |\n"
    "| **total** | | **~$4.87** |"
)

# PISA-discovered seed (ETF winner, branching init, nodes=5), ratio 1.98.
# NOTE: an earlier version of this file had these values hand-typed with
# fabricated extra decimal digits after printing them rounded for
# readability -- that silently changed the instance and gave a benchmark
# result contradicting the manual sweep used to pick PERTURBATION. These are
# read directly (full precision, no rounding) from
# outputs/ETF_vs_CPoP/seeds/ETF_vs_CPoP_{network,taskgraph}.json.
_BASE_NODES = {'3': 1.0, '2': 1.0, '0': 1.0, '1': 1.0, '4': 1.0}
_BASE_EDGES = {('0', '1'): 0.29703075101583487, ('0', '4'): 1.0, ('1', '4'): 0.4726080197716487, ('2', '4'): 0.1, ('2', '3'): 0.35109150676249906, ('0', '3'): 0.9273299352855133, ('1', '3'): 0.755064525012451, ('3', '4'): 0.7865614983436038, ('0', '2'): 0.2103189942524325, ('1', '2'): 0.24639932185599078}
_BASE_TASKS = {'7': 0.7222110090003764, '3': 0.1, '2': 0.1, '0': 0.5304079038080186, '1': 0.34630994072175547, '4': 0.8169380192269069, '6': 0.30706709606541027, '5': 0.22605011136697026}
_BASE_DEPS = {('1', '6'): 0.7899922737003527, ('2', '6'): 0.8416905059975791, ('3', '2'): 0.3603735833048211, ('1', '4'): 0.844437493504839, ('7', '3'): 0.7796770716567794, ('6', '5'): 0.5796176297332504, ('1', '5'): 1.0, ('3', '5'): 0.7724443428324956}

PERTURBATION = 0.02  # tightest radius tried; even here geomean is only ~1.3, p10<1


def make_instance(rng: random.Random) -> Tuple[Network, TaskGraph]:
    """Draw one random instance: a tight joint perturbation of the PISA seed."""
    nodes = [(name, speed) for name, speed in _BASE_NODES.items()]
    edges = [(s, t, speed) for (s, t), speed in _BASE_EDGES.items()]
    network = Network.create(nodes=nodes, edges=edges)

    tasks = [(name, cost) for name, cost in _BASE_TASKS.items()]
    deps = [(s, t, size) for (s, t), size in _BASE_DEPS.items()]
    task_graph = TaskGraph.create(tasks=tasks, dependencies=deps)

    return perturb_instance(network, task_graph, PERTURBATION, rng)


if __name__ == "__main__":
    from family_lib import evaluate_family, format_report

    stats = evaluate_family(make_instance, WINNER, LOSER, n=300, rng=random.Random(0))
    print(HYPOTHESIS)
    print(format_report(stats))
