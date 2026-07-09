"""PISA-style adversarial search for a CTG where CHEFT beats HEFT.

SAGA's PISA (saga.pisa) does simulated-annealing over (task_graph, network) to
maximize energy = scheduler.makespan / base_scheduler.makespan. Two things make it
unsuitable here directly: it uses the plain schedule makespan (not the expected
makespan over traces that matters for conditional scheduling), and its change
operators produce plain graphs (dropping conditional probabilities). This script
keeps PISA's idea but with a conditional-aware energy and structure-preserving moves.

State: a conditional task graph + network, represented as plain dicts so moves are
easy; rebuilt into saga objects for each evaluation.
Energy (maximized): E[makespan] of HEFT / E[makespan] of CHEFT over the trace
distribution. Energy > 1 means CHEFT (reweighted) beats HEFT (unweighted).
Moves preserve graph structure and the trace set; only weights/speeds/branch
probabilities change.
"""

import json
import math
import pathlib
import random

from saga import Network
from saga.conditional import (
    ConditionalTaskGraph,
    ConditionalTaskGraphEdge,
    recalculate_trace_times,
)
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
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
HEFT = ParametricScheduler(UpwardRanking(), GreedyInsert(compare=_EFT))
CHEFT = ParametricScheduler(
    ProbabilityWeighted(base=UpwardRanking()), GreedyInsert(compare=_EFT)
)


# --------------------------------------------------------------------------- #
# State <-> saga objects
# --------------------------------------------------------------------------- #


def state_from_random(rng: random.Random, levels: int, bf: int, num_nodes: int) -> dict:
    """Build an initial state dict from a random conditional branching DAG."""
    import numpy as np

    np.random.seed(rng.randint(0, 2**31 - 1))
    ctg = get_random_conditional_branching_dag(
        levels=levels, branching_factor=bf, conditional_parent_probability=0.6
    )
    net = get_network(num_nodes)
    tasks = {t.name: t.cost for t in ctg.tasks}
    edges = [
        [e.source, e.target, e.size,
         e.probability if isinstance(e, ConditionalTaskGraphEdge) else 1.0]
        for e in ctg.dependencies
    ]
    node_speeds = {n.name: n.speed for n in net.nodes}
    edge_speeds = {}
    for e in net.edges:
        if e.source != e.target:
            edge_speeds[tuple(sorted((e.source, e.target)))] = e.speed
    return {"tasks": tasks, "edges": edges, "node_speeds": node_speeds, "edge_speeds": edge_speeds}


def build(state: dict) -> tuple[Network, ConditionalTaskGraph]:
    ctg = ConditionalTaskGraph.create(
        tasks=[(n, c) for n, c in state["tasks"].items()],
        dependencies=[
            ConditionalTaskGraphEdge(source=s, target=t, size=sz, probability=p)
            for s, t, sz, p in state["edges"]
        ],
    )
    net = Network.create(
        nodes=[(n, s) for n, s in state["node_speeds"].items()],
        edges=[(u, v, s) for (u, v), s in state["edge_speeds"].items()],
    )
    return net, ctg


# --------------------------------------------------------------------------- #
# Energy = E[HEFT makespan] / E[CHEFT makespan]
# --------------------------------------------------------------------------- #


def _realized_makespan(schedule, trace_tasks) -> float:
    task_set = set(trace_tasks)
    mapping = {
        node: [t for t in tasks if t.name in task_set]
        for node, tasks in schedule.mapping.items()
    }
    recalced = recalculate_trace_times(mapping, schedule)
    return max((t.end for tasks in recalced.values() for t in tasks), default=0.0)


def expected_makespans(state: dict) -> tuple[float, float, list]:
    net, ctg = build(state)
    traces = ctg.identify_traces_detailed()
    heft = CHEFT_ = 0.0
    s_heft = HEFT.schedule(net, ctg)
    s_cheft = CHEFT.schedule(net, ctg)
    for tr in traces:
        p = tr["probability"]
        heft += p * _realized_makespan(s_heft, tr["tasks"])
        CHEFT_ += p * _realized_makespan(s_cheft, tr["tasks"])
    return heft, CHEFT_, traces


def energy(state: dict) -> float:
    heft, cheft, _ = expected_makespans(state)
    return heft / cheft if cheft > 0 else 1.0


# --------------------------------------------------------------------------- #
# Structure-preserving moves
# --------------------------------------------------------------------------- #


def _conditional_forks(state: dict) -> dict:
    """parent -> list of edge-indices, for parents whose out-edges are a conditional fork."""
    forks: dict = {}
    by_parent: dict = {}
    for i, (s, t, sz, p) in enumerate(state["edges"]):
        by_parent.setdefault(s, []).append(i)
    for parent, idxs in by_parent.items():
        if len(idxs) > 1 and any(state["edges"][i][3] < 1.0 for i in idxs):
            forks[parent] = idxs
    return forks


# Bounds keep found instances realistic (the SA would otherwise exploit extreme
# weight ratios to inflate the ratio into a numerical artifact).
COST_RANGE = (0.5, 10.0)
SIZE_RANGE = (0.5, 10.0)
SPEED_RANGE = (0.25, 2.0)


def neighbor(state: dict, rng: random.Random) -> dict:
    new = {
        "tasks": dict(state["tasks"]),
        "edges": [list(e) for e in state["edges"]],
        "node_speeds": dict(state["node_speeds"]),
        "edge_speeds": dict(state["edge_speeds"]),
    }
    move = rng.choice(["task", "edge", "node", "link", "prob"])

    def jitter(x, lo, hi):
        return min(hi, max(lo, x * rng.uniform(0.6, 1.6)))

    if move == "task":
        k = rng.choice(list(new["tasks"]))
        new["tasks"][k] = jitter(new["tasks"][k], *COST_RANGE)
    elif move == "edge":
        i = rng.randrange(len(new["edges"]))
        new["edges"][i][2] = jitter(new["edges"][i][2], *SIZE_RANGE)
    elif move == "node":
        k = rng.choice(list(new["node_speeds"]))
        new["node_speeds"][k] = jitter(new["node_speeds"][k], *SPEED_RANGE)
    elif move == "link" and new["edge_speeds"]:
        k = rng.choice(list(new["edge_speeds"]))
        new["edge_speeds"][k] = jitter(new["edge_speeds"][k], *SPEED_RANGE)
    elif move == "prob":
        forks = _conditional_forks(new)
        if forks:
            idxs = forks[rng.choice(list(forks))]
            probs = [max(0.05, new["edges"][i][3] + rng.uniform(-0.3, 0.3)) for i in idxs]
            total = sum(probs)
            for i, p in zip(idxs, probs):
                new["edges"][i][3] = p / total
    return new


# --------------------------------------------------------------------------- #
# Simulated annealing
# --------------------------------------------------------------------------- #


def anneal(state: dict, rng: random.Random, iters: int, t0: float, cooling: float):
    cur_e = energy(state)
    best_state, best_e = state, cur_e
    temp = t0
    for _ in range(iters):
        cand = neighbor(state, rng)
        cand_e = energy(cand)
        delta = cand_e - cur_e
        if delta > 0 or rng.random() < math.exp(delta / max(temp, 1e-9)):
            state, cur_e = cand, cand_e
            if cur_e > best_e:
                best_state, best_e = state, cur_e
        temp *= cooling
    return best_state, best_e


def report(state: dict) -> None:
    """Print the instance and the per-trace makespan breakdown for HEFT vs CHEFT."""
    net, ctg = build(state)
    traces = ctg.identify_traces_detailed()
    s_heft = HEFT.schedule(net, ctg)
    s_cheft = CHEFT.schedule(net, ctg)
    heft = cheft = 0.0
    print(f"Tasks (name: cost): {{" + ", ".join(f'{n}:{c:.2f}' for n, c in state['tasks'].items()) + "}")
    print(f"Nodes (name: speed): {{" + ", ".join(f'{n}:{s:.2f}' for n, s in state['node_speeds'].items()) + "}")
    print("Per-trace realized makespan:")
    print(f"  {'trace':<22} {'p':>5}  {'HEFT':>7} {'CHEFT':>7}")
    for tr in traces:
        p = tr["probability"]
        mh = _realized_makespan(s_heft, tr["tasks"])
        mc = _realized_makespan(s_cheft, tr["tasks"])
        heft += p * mh
        cheft += p * mc
        print(f"  {tr['name']:<22} {p:>5.2f}  {mh:>7.3f} {mc:>7.3f}  {'<- CHEFT better' if mc < mh - 1e-6 else ''}")
    print(f"  {'E[makespan]':<22} {'':>5}  {heft:>7.3f} {cheft:>7.3f}   (CHEFT better by {(heft/cheft-1)*100:.1f}%)")


def main() -> None:
    rng = random.Random(0)
    best_state, best_e = None, -math.inf
    restarts = 6
    for r in range(restarts):
        init = state_from_random(rng, levels=3, bf=2, num_nodes=3)
        s, e = anneal(init, rng, iters=1000, t0=0.05, cooling=0.995)
        print(f"restart {r}: best energy (E[HEFT]/E[CHEFT]) = {e:.4f}", flush=True)
        if e > best_e:
            best_state, best_e = s, e

    assert best_state is not None
    print(f"\n=== BEST INSTANCE: CHEFT beats HEFT by {(best_e-1)*100:.1f}% (energy {best_e:.4f}) ===")
    report(best_state)
    # JSON can't key on tuples; encode network edges as "u|v".
    serializable = dict(best_state)
    serializable["edge_speeds"] = {f"{u}|{v}": s for (u, v), s in best_state["edge_speeds"].items()}
    (outdir / "pisa_best_instance.json").write_text(json.dumps(serializable, indent=2))
    print(f"\nSaved best instance to {outdir / 'pisa_best_instance.json'}")


if __name__ == "__main__":
    main()
