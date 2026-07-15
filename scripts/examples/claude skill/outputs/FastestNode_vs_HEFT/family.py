"""Scheduler-adversarial family: FastestNode (WINNER) beats HEFT (LOSER).

Mechanism (see HYPOTHESIS): a fork-join DAG on a network with one dominant fast
node, a few only-slightly-slower nodes, and VERY slow (low-bandwidth) links.
The fork splits into a small number (2-3) of parallel "middle" tasks that each
carry heavy data into a single join task.

HEFT schedules greedily in rank order and is myopic: when it places middle task
m_k, the fast node is already busy running m_0, and a slightly-slower node
offers an earlier finish time for m_k, so HEFT offloads m_k there. That decision
only accounts for m_k's own finish, not the future. At the join, every scattered
middle result must be shipped back over the very slow links, and that gather
cost dwarfs the little parallelism HEFT bought. FastestNode instead piles every
task onto the single fastest node, so all communication is intra-node (free);
its makespan is just sum(task costs) / fastest_speed, and it wins handily.

Two design choices make the trap reliable:
  * slow nodes are only slightly slower than the fast node (0.65-0.90 vs ~1.0),
    so offloading a middle task genuinely looks attractive to HEFT's EFT rule;
  * inter-node links are very slow (0.03-0.20), so the heavy join gather is
    brutal once tasks have been scattered.

Discovered with seed_pisa.py (chain init, ratio ~2.1) and then sharpened by
identifying HEFT's fork-join myopia as the true lever.
"""

from __future__ import annotations

import random
from typing import Tuple

from saga import Network, TaskGraph

WINNER = "FastestNode"   # expected low makespan
LOSER = "HEFT"           # expected high makespan

HYPOTHESIS = (
    "A fork-join DAG with heavy join data on a network of one fast node, a few "
    "only-slightly-slower nodes, and very slow links baits HEFT into offloading "
    "the parallel middle tasks onto slower nodes to shave each task's finish "
    "time. HEFT's greedy per-task rule ignores the future gather, so at the join "
    "it must ship every scattered result back over the slow links. FastestNode "
    "keeps everything on the fastest node, making all communication free, and "
    "wins because communication dominates once the tasks are scattered."
)


def make_instance(rng: random.Random) -> Tuple[Network, TaskGraph]:
    """Draw one random instance from the family."""
    # --- network: one fast node, a few tempting-but-slower nodes, slow links --
    num_nodes = rng.randint(4, 5)
    nodes = [("0", rng.uniform(0.95, 1.0))]              # the clear fastest node
    for i in range(1, num_nodes):
        # Only slightly slower => offloading looks attractive to HEFT.
        nodes.append((str(i), rng.uniform(0.65, 0.90)))
    # Fully connected UNDIRECTED network (add each pair once). Very low
    # bandwidth => any cross-node data transfer is very expensive.
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edges.append((str(i), str(j), rng.uniform(0.03, 0.20)))
    network = Network.create(nodes=nodes, edges=edges)

    # --- task graph: fork -> a few heavy-output middle tasks -> join ----------
    width = rng.randint(2, 3)          # narrow: just enough to tempt scattering
    tasks = [("fork", rng.uniform(0.1, 0.3))]
    deps = []
    mids = []
    for k in range(width):
        name = f"m{k}"
        tasks.append((name, rng.uniform(0.6, 1.0)))
        mids.append(name)
        # Cheap fork->middle data so the middle tasks are cheap to distribute...
        deps.append(("fork", name, rng.uniform(0.01, 0.05)))
    tasks.append(("join", rng.uniform(0.1, 0.3)))
    for m in mids:
        # ...but heavy middle->join data so gathering scattered results is brutal.
        deps.append((m, "join", rng.uniform(0.85, 1.2)))
    task_graph = TaskGraph.create(tasks=tasks, dependencies=deps)

    return network, task_graph


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]
                          / ".claude/skills/find-scheduler-family/scripts"))
    from family_lib import evaluate_family, format_report

    stats = evaluate_family(make_instance, WINNER, LOSER, n=200, rng=random.Random(0))
    print(HYPOTHESIS)
    print(format_report(stats))
