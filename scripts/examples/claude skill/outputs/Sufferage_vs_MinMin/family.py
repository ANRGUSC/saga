"""Scheduler-adversarial family: Sufferage (WINNER) beats MinMin (LOSER).

Mechanism (see HYPOTHESIS): MinMin (saga/schedulers/minmin.py) schedules,
each round, whichever single (available task, node) pair has the globally
lowest estimated completion time (ECT) -- with no notion of urgency. It can
let a batch of flexible tasks (roughly equally happy on any node) repeatedly
grab a node that one OTHER task needs far more, simply because their raw ECT
number happens to be lowest that round.

Sufferage (saga/schedulers/sufferage.py) instead computes, per available
task, "sufferage" = (2nd-best ECT - best ECT): how much a task would lose if
denied its best node. It always schedules the task with the LARGEST
sufferage first, protecting tasks whose best option is dramatically better
than their alternatives -- exactly the tasks MinMin's rule can starve.

Discovered with seed_pisa.py (branching init, nodes=5, ratio 2.59): a mixed
DAG with an isolated "flexible" task, an independent root feeding a shared
sink, and a several-task chain feeding the same sink -- giving the sink task
a strong data-locality tie to one predecessor's node (high sufferage) while
other ready tasks are comparatively indifferent to node choice (low
sufferage). Generalized via TIGHT joint perturbation of the exact seed
(family_lib.perturb_instance), not independent resampling of its marginal
ranges -- see SKILL.md step 3. A perturbation-radius sweep showed this gap
degrades gracefully (unlike HEFT vs CPoP's sharp cliff): geomean ~2.0-2.05
and p10 ~1.37 hold from frac=0.02 up through about frac=0.10, only fully
collapsing (and even reversing) out past frac~0.5-1.0. frac=0.02 is used
here for a clean safety margin above the STRONG bar.
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

WINNER = "Sufferage"  # expected low makespan
LOSER = "MinMin"      # expected high makespan

HYPOTHESIS = (
    "MinMin schedules, each round, whichever single (task, node) pair has "
    "the globally lowest estimated completion time, with no notion of "
    "urgency -- so a batch of flexible tasks (roughly equally happy on any "
    "node) can repeatedly grab a node that some OTHER task needs far more "
    "badly, just because their raw completion-time number happens to be "
    "lowest that round. Sufferage instead always schedules whichever "
    "available task has the largest gap between its best and second-best "
    "node (its 'sufferage') first, protecting exactly the tasks MinMin's "
    "rule can starve -- here, a sink task with a strong data-locality tie "
    "to one predecessor's node, competing against more flexible ready tasks "
    "that don't care much which node they land on."
)

CLAUDE_COST_ESTIMATE = (
    "Exact, read directly from the local Claude Code session transcript "
    "(`~/.claude/projects/.../*.jsonl`) for the turns spanning this specific "
    "investigation (from the skill invocation through the next pair's "
    "invocation). Sonnet 5, current intro pricing (through 2026-08-31): "
    "$2.00/$10.00 per 1M input/output tokens, cache write (1h TTL) $4.00/1M, "
    "cache read $0.20/1M.\n\n"
    "| | tokens | cost |\n"
    "|---|---:|---:|\n"
    "| input (uncached) | 62 | $0.00 |\n"
    "| output | 33,679 | $0.34 |\n"
    "| cache write (1h) | 67,703 | $0.27 |\n"
    "| cache read | 7,220,987 | $1.44 |\n"
    "| **total** | | **~$2.05** |"
)

# Exact PISA-discovered seed (outputs/Sufferage_vs_MinMin/seeds/), ratio 2.59.
_BASE_NODES = {"1": 0.9709414395213729, "2": 0.34929869777905126,
               "0": 0.4795329932277726, "4": 0.1, "3": 0.4188342871508429}
_BASE_EDGES = {("1", "2"): 0.7635677238146025, ("3", "4"): 0.7674443925744551,
               ("0", "3"): 0.7014165932332726, ("0", "2"): 0.1,
               ("1", "3"): 0.5657509337726853, ("2", "4"): 0.1,
               ("0", "4"): 0.8599422535654766, ("1", "4"): 0.6874274618473099,
               ("2", "3"): 0.1, ("0", "1"): 0.1}
_BASE_TASKS = {"1": 0.6174129557712829, "7": 1.0, "6": 0.44870725729292486,
               "2": 0.5311918584709301, "0": 0.19295221980762047,
               "5": 0.6025560592712444, "3": 0.7274046444570887,
               "4": 0.19601937527638966}
_BASE_DEPS = {("1", "2"): 0.17969049917379633, ("7", "5"): 0.41102931114756577,
              ("0", "5"): 0.6935705570409186, ("4", "3"): 0.6529571893904953,
              ("3", "7"): 0.5780462461362174, ("1", "4"): 0.6319847940333038}

PERTURBATION = 0.02  # +-2%; verified to clear geomean>=2.0 and p10>=1.2 with margin


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
