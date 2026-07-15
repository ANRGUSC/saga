"""Shared helpers for the find-scheduler-family skill.

A "problem instance" is a (Network, TaskGraph) pair. A "family" is a callable
`make_instance(rng: random.Random) -> tuple[Network, TaskGraph]` that draws a
random instance from some distribution. This module provides the tools to
resolve schedulers by name, summarize an instance's structure, and estimate the
expected makespan ratio between two schedulers over a family.
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Callable, Dict, List, Tuple

import networkx as nx

# SAGA emits benign "multiple sinks; adding super sink" warnings on the root
# logger for many valid instances. Quiet them so benchmark output stays readable.
logging.getLogger().setLevel(logging.ERROR)

from saga import Network, TaskGraph, Scheduler
from saga.pisa import SCHEDULERS

Instance = Tuple[Network, TaskGraph]
MakeInstance = Callable[..., Instance]

SUPER_NODES = {"__super_source__", "__super_sink__"}


def resolve_scheduler(name: str) -> Scheduler:
    """Resolve a scheduler by its PISA-registry name (e.g. 'HEFT', 'FastestNode')."""
    if name not in SCHEDULERS:
        valid = ", ".join(sorted(SCHEDULERS.keys()))
        raise KeyError(f"Unknown scheduler {name!r}. Valid names: {valid}")
    return SCHEDULERS[name]


def makespan(scheduler: Scheduler, network: Network, task_graph: TaskGraph) -> float:
    """Schedule and return the makespan."""
    return scheduler.schedule(network, task_graph).makespan


def estimate_ccr(network: Network, task_graph: TaskGraph) -> float:
    """Communication-to-computation ratio, matching Network.scale_to_ccr's definition."""
    real_deps = [
        d for d in task_graph.dependencies
        if d.source not in SUPER_NODES and d.target not in SUPER_NODES
    ]
    real_tasks = [t for t in task_graph.tasks if t.name not in SUPER_NODES]
    if not real_deps or not real_tasks or not network.nodes or not network.edges:
        return 0.0
    avg_node_speed = sum(n.speed for n in network.nodes) / len(network.nodes)
    avg_edge_speed = sum(e.speed for e in network.edges) / len(network.edges)
    avg_task_cost = sum(t.cost for t in real_tasks) / len(real_tasks)
    avg_data_size = sum(d.size for d in real_deps) / len(real_deps)
    avg_comp_time = avg_task_cost / avg_node_speed
    avg_comm_time = avg_data_size / avg_edge_speed
    return avg_comm_time / avg_comp_time if avg_comp_time else 0.0


def summarize_instance(network: Network, task_graph: TaskGraph) -> Dict:
    """Structural fingerprint of an instance, used to reverse-engineer a family."""
    g: nx.DiGraph = task_graph.graph
    real_tasks = [t for t in task_graph.tasks if t.name not in SUPER_NODES]
    real_deps = [
        d for d in task_graph.dependencies
        if d.source not in SUPER_NODES and d.target not in SUPER_NODES
    ]
    node_speeds = [n.speed for n in network.nodes]
    edge_speeds = [e.speed for e in network.edges if e.source != e.target]
    task_costs = [t.cost for t in real_tasks]
    dep_sizes = [d.size for d in real_deps]

    # width per topological level (longest-path depth)
    depth: Dict[str, int] = {}
    for n in nx.topological_sort(g):
        preds = list(g.predecessors(n))
        depth[n] = 0 if not preds else 1 + max(depth[p] for p in preds)
    level_width: Dict[int, int] = {}
    for n, d in depth.items():
        if n not in SUPER_NODES:
            level_width[d] = level_width.get(d, 0) + 1

    def rng(xs: List[float]) -> Tuple[float, float]:
        return (round(min(xs), 4), round(max(xs), 4)) if xs else (0.0, 0.0)

    try:
        crit_len = nx.dag_longest_path_length(g)
    except Exception:
        crit_len = None

    return {
        "num_nodes": len(network.nodes),
        "num_tasks": len(real_tasks),
        "num_deps": len(real_deps),
        "node_speed_range": rng(node_speeds),
        "edge_speed_range": rng(edge_speeds),
        "task_cost_range": rng(task_costs),
        "dep_size_range": rng(dep_sizes),
        "node_speed_spread": round(max(node_speeds) / min(node_speeds), 3) if node_speeds and min(node_speeds) else None,
        "edge_speed_spread": round(max(edge_speeds) / min(edge_speeds), 3) if edge_speeds and min(edge_speeds) else None,
        "ccr": round(estimate_ccr(network, task_graph), 4),
        "critical_path_len": crit_len,
        "max_level_width": max(level_width.values()) if level_width else 0,
        "num_levels": len(level_width),
    }


def evaluate_family(
    make_instance: MakeInstance,
    winner: str,
    loser: str,
    n: int,
    rng,
    threshold: float = 2.0,
) -> Dict:
    """Sample n instances and measure makespan(loser) / makespan(winner).

    A large ratio means `winner` dramatically outperforms `loser`. Returns
    summary statistics plus the raw ratios, and counts of degenerate samples.
    """
    win = resolve_scheduler(winner)
    lose = resolve_scheduler(loser)
    ratios: List[float] = []
    win_ms: List[float] = []
    lose_ms: List[float] = []
    errors = 0
    for _ in range(n):
        try:
            network, task_graph = make_instance(rng)
            w = makespan(win, network, task_graph)
            l = makespan(lose, network, task_graph)
            if w <= 0:
                errors += 1
                continue
            ratios.append(l / w)
            win_ms.append(w)
            lose_ms.append(l)
        except Exception:
            errors += 1
    if not ratios:
        return {"n": n, "errors": errors, "usable": 0, "ratios": []}

    geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    srt = sorted(ratios)
    def pct(p: float) -> float:
        return srt[min(len(srt) - 1, max(0, int(round(p * (len(srt) - 1)))))]
    return {
        "winner": winner,
        "loser": loser,
        "n": n,
        "usable": len(ratios),
        "errors": errors,
        "mean_ratio": statistics.mean(ratios),
        "geomean_ratio": geomean,
        "median_ratio": statistics.median(ratios),
        "p10_ratio": pct(0.10),
        "p90_ratio": pct(0.90),
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
        "stdev_ratio": statistics.pstdev(ratios),
        "frac_above_threshold": sum(1 for r in ratios if r >= threshold) / len(ratios),
        "threshold": threshold,
        "mean_winner_makespan": statistics.mean(win_ms),
        "mean_loser_makespan": statistics.mean(lose_ms),
        "ratios": ratios,
    }


def perturb_instance(network: Network, task_graph: TaskGraph, frac: float, rng) -> Instance:
    """Jointly perturb every weight in a seed instance by +-frac (relative).

    Keeps the exact topology (same tasks, dependencies, nodes, edges) and only
    resamples magnitudes, each within `value * (1 +- frac)`. This preserves
    whatever relative orderings among rank/EFT comparisons made the seed
    adversarial. Independently redrawing each parameter across its full
    marginal range (e.g. from `*_summary.json`'s reported min/max) usually
    destroys that joint alignment even though every individual draw is
    "in range" -- see SKILL.md step 3. Start generalization here, not there.
    """
    def bump(v: float) -> float:
        return max(1e-6, rng.uniform(v * (1 - frac), v * (1 + frac)))

    nodes = [(n.name, bump(n.speed)) for n in network.nodes]
    edges = [(e.source, e.target, bump(e.speed)) for e in network.edges if e.source != e.target]
    new_network = Network.create(nodes=nodes, edges=edges)

    real_tasks = [t for t in task_graph.tasks if t.name not in SUPER_NODES]
    real_deps = [
        d for d in task_graph.dependencies
        if d.source not in SUPER_NODES and d.target not in SUPER_NODES
    ]
    tasks = [(t.name, bump(t.cost)) for t in real_tasks]
    deps = [(d.source, d.target, bump(d.size)) for d in real_deps]
    new_task_graph = TaskGraph.create(tasks=tasks, dependencies=deps)

    return new_network, new_task_graph


def sweep_perturbation(
    network: Network,
    task_graph: TaskGraph,
    winner: str,
    loser: str,
    rng,
    fracs: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0),
    n: int = 300,
    threshold: float = 2.0,
) -> Dict[float, Dict]:
    """Sweep joint-perturbation radius around one seed instance.

    Run this on the best PISA seed BEFORE attempting a broad, freely
    re-randomized family. It reveals whether the seed's gap survives numeric
    perturbation at all, and if so, how wide a radius stays safe:
      - Effect survives out past ~50-100% frac -> a CATEGORICAL mechanism
        (the loser ignores some information altogether, e.g. FastestNode
        ignoring parallelism). A broad structural family is safe.
      - Effect only survives to ~10-20% frac -> a COMPARISON-FRAGILE
        mechanism (the gap depends on a delicate rank/EFT tie-break, e.g.
        HEFT vs CPoP). The family IS the tight perturbation ball at a radius
        just inside the observed cliff -- that's a legitimate, reportable
        family, not a lesser result.
      - Effect collapses even at the smallest frac (~2%) -> likely a
        numeric coincidence in the seed itself, not a real gap. Only NOW is
        NO EFFECT a fair conclusion -- see SKILL.md's decision rule.
    """
    results: Dict[float, Dict] = {}
    for frac in fracs:
        def make(rng_, _frac=frac):
            return perturb_instance(network, task_graph, _frac, rng_)
        results[frac] = evaluate_family(make, winner, loser, n=n, rng=rng, threshold=threshold)
    return results


def format_perturbation_sweep(results: Dict[float, Dict]) -> str:
    """One-line-per-radius summary of a perturbation sweep."""
    lines = [
        "perturbation sweep (joint +-frac around the seed):",
        f"  {'frac':>6}  {'geomean':>8}  {'median':>8}  {'p10':>6}  {'frac>=thr':>10}",
    ]
    for frac, stats in sorted(results.items()):
        if not stats.get("usable"):
            lines.append(f"  {frac:>6.2f}  (no usable samples)")
            continue
        lines.append(
            f"  {frac:>6.2f}  {stats['geomean_ratio']:>8.3f}  {stats['median_ratio']:>8.3f}  "
            f"{stats['p10_ratio']:>6.3f}  {stats['frac_above_threshold']:>9.1%}"
        )
    return "\n".join(lines)


def save_perturbation_sweep_plot(results: Dict[float, Dict], path) -> None:
    """Plot geomean and p10 ratio vs perturbation radius -- shows the cliff."""
    plt = _lazy_plt()
    fracs = sorted(results)
    geomeans = [results[f].get("geomean_ratio", float("nan")) for f in fracs]
    p10s = [results[f].get("p10_ratio", float("nan")) for f in fracs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fracs, geomeans, "o-", color="#55A868", label="geomean ratio")
    ax.plot(fracs, p10s, "o-", color="#C44E52", label="p10 ratio")
    ax.axhline(1.0, color="#888", ls=":", lw=1, label="ratio = 1 (tie)")
    ax.set_xlabel("joint perturbation radius (+-frac around seed)")
    ax.set_ylabel("makespan ratio loser/winner")
    ax.set_title("Perturbation sweep: where does the gap collapse?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _lazy_plt():
    """Import matplotlib with a headless backend (safe under Bash/no-display)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_ratio_histogram(stats: Dict, path, ) -> None:
    """Histogram of the makespan-ratio distribution with mean/median/threshold lines."""
    plt = _lazy_plt()
    ratios = stats["ratios"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=min(40, max(10, len(ratios) // 8)), color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.axvline(1.0, color="#888", ls=":", lw=1.5, label="ratio = 1 (tie)")
    ax.axvline(stats["threshold"], color="#C44E52", ls="--", lw=1.5, label=f"threshold = {stats['threshold']}")
    ax.axvline(stats["geomean_ratio"], color="#55A868", ls="-", lw=2, label=f"geomean = {stats['geomean_ratio']:.2f}")
    ax.axvline(stats["median_ratio"], color="#DD8452", ls="-", lw=1.5, label=f"median = {stats['median_ratio']:.2f}")
    ax.set_xlabel(f"makespan({stats['loser']}) / makespan({stats['winner']})")
    ax.set_ylabel("count")
    ax.set_title(f"Ratio distribution over {stats['usable']} instances\n"
                 f"{stats['winner']} (winner) vs {stats['loser']} (loser)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_instance_figures(network: Network, task_graph: TaskGraph, winner: str, loser: str, outdir) -> Dict:
    """Draw an exemplar instance: task graph, network, and a Gantt per scheduler.

    Returns a dict with the two makespans and the saved file paths.
    """
    import pathlib
    from saga.utils.draw import draw_task_graph, draw_network, draw_gantt
    plt = _lazy_plt()
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    draw_task_graph(task_graph.graph, axis=ax)
    ax.set_title("Task graph (node=cost, edge=data size)")
    fig.tight_layout(); fig.savefig(outdir / "exemplar_task_graph.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    draw_network(network.graph, axis=ax)
    ax.set_title("Network (node=compute speed, edge=bandwidth)")
    fig.tight_layout(); fig.savefig(outdir / "exemplar_network.png", dpi=120); plt.close(fig)

    win_s = resolve_scheduler(winner).schedule(network, task_graph)
    lose_s = resolve_scheduler(loser).schedule(network, task_graph)
    xmax = max(win_s.makespan, lose_s.makespan) * 1.05
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    draw_gantt(win_s.mapping, axis=axes[0], xmax=xmax)
    axes[0].set_title(f"{winner} (winner) — makespan {win_s.makespan:.3f}")
    draw_gantt(lose_s.mapping, axis=axes[1], xmax=xmax)
    axes[1].set_title(f"{loser} (loser) — makespan {lose_s.makespan:.3f}")
    fig.tight_layout(); fig.savefig(outdir / "exemplar_gantt.png", dpi=120); plt.close(fig)

    return {
        "winner_makespan": win_s.makespan,
        "loser_makespan": lose_s.makespan,
        "ratio": lose_s.makespan / win_s.makespan if win_s.makespan else float("inf"),
        "files": ["exemplar_task_graph.png", "exemplar_network.png", "exemplar_gantt.png"],
    }


def pick_exemplar(make_instance, winner: str, loser: str, rng, n: int, mode: str = "median"):
    """Sample n instances and return the (network, task_graph, ratio) whose ratio is
    closest to the median (mode='median', a representative case) or the maximum
    (mode='max', the most dramatic case)."""
    win = resolve_scheduler(winner)
    lose = resolve_scheduler(loser)
    cand = []
    for _ in range(n):
        try:
            net, tg = make_instance(rng)
            w = makespan(win, net, tg)
            if w <= 0:
                continue
            cand.append((net, tg, makespan(lose, net, tg) / w))
        except Exception:
            continue
    if not cand:
        return None
    cand.sort(key=lambda c: c[2])
    if mode == "max":
        return cand[-1]
    return cand[len(cand) // 2]


def save_ccr_sweep(make_instance, winner: str, loser: str, rng, ccrs, n, path) -> Dict:
    """Sweep target CCR, rescaling each sampled network, and plot mean ratio vs CCR.

    Requires the family's instances to have communication (dependency sizes > 0);
    for zero-comm families CCR is undefined and this is skipped by the caller.
    Returns {ccr: mean_ratio}.
    """
    plt = _lazy_plt()
    win = resolve_scheduler(winner)
    lose = resolve_scheduler(loser)
    # Draw a fixed pool so each CCR sees the same base instances.
    pool = []
    for _ in range(n):
        try:
            pool.append(make_instance(rng))
        except Exception:
            pass
    means = {}
    for ccr in ccrs:
        rs = []
        for net, tg in pool:
            try:
                scaled = net.scale_to_ccr(tg, target_ccr=ccr)
                w = makespan(win, scaled, tg)
                if w > 0:
                    rs.append(makespan(lose, scaled, tg) / w)
            except Exception:
                pass
        if rs:
            means[ccr] = math.exp(sum(math.log(r) for r in rs) / len(rs))
    if means:
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = sorted(means)
        ax.plot(xs, [means[x] for x in xs], "o-", color="#4C72B0")
        ax.axhline(1.0, color="#888", ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("target CCR (communication-to-computation ratio, log scale)")
        ax.set_ylabel(f"geomean makespan({loser})/makespan({winner})")
        ax.set_title("Adversarial gap vs CCR")
        fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return means


def format_report(stats: Dict) -> str:
    """Human-readable one-screen report from evaluate_family output."""
    if not stats.get("usable"):
        return f"No usable samples (n={stats.get('n')}, errors={stats.get('errors')})."
    lines = [
        f"Family: {stats['winner']} (winner) vs {stats['loser']} (loser)",
        f"  samples: {stats['usable']}/{stats['n']} usable ({stats['errors']} errors)",
        f"  makespan ratio loser/winner:",
        f"    mean    {stats['mean_ratio']:.3f}   geomean {stats['geomean_ratio']:.3f}",
        f"    median  {stats['median_ratio']:.3f}   stdev   {stats['stdev_ratio']:.3f}",
        f"    p10     {stats['p10_ratio']:.3f}   p90     {stats['p90_ratio']:.3f}",
        f"    min     {stats['min_ratio']:.3f}   max     {stats['max_ratio']:.3f}",
        f"  fraction with ratio >= {stats['threshold']}: {stats['frac_above_threshold']:.1%}",
        f"  mean makespan: winner={stats['mean_winner_makespan']:.3f}  loser={stats['mean_loser_makespan']:.3f}",
    ]
    verdict = classify_verdict(stats)
    lines.append(f"  verdict: {verdict}")
    return "\n".join(lines)


def classify_verdict(stats: Dict) -> str:
    """Classify a family's strength from evaluate_family's stats.

    Four tiers, in order:
      - STRONG: geomean >= threshold AND p10 >= 1.2 -- large AND consistent.
      - MODERATE/CONSISTENT: mean >= 1.2 AND p10 >= 1.2 -- doesn't clear the
        threshold magnitude, but the effect holds up even at the 10th
        percentile, so it's real and reliable, just smaller. A tight,
        low-variance ~1.9x effect belongs here, not lumped in with a
        volatile one.
      - WEAK/INCONSISTENT: mean >= 1.2 but p10 < 1.2 -- there's a real
        average lean, but a meaningful fraction of samples don't show it
        (or even reverse it). This is the "not broad, not reliable" case
        step 5 (iterate) is for.
      - NO EFFECT: mean < 1.2.

    p10 (not stdev) is the consistency signal: it directly answers "does
    the effect hold up for the unlucky tail of the distribution," which is
    what actually matters for a broad family, whereas raw stdev conflates
    a wide-but-still-always-winning distribution with a narrow-but-
    sometimes-losing one.
    """
    if stats["geomean_ratio"] >= stats["threshold"] and stats["p10_ratio"] >= 1.2:
        return "STRONG"
    if stats["mean_ratio"] >= 1.2 and stats["p10_ratio"] >= 1.2:
        return "MODERATE/CONSISTENT"
    if stats["mean_ratio"] >= 1.2:
        return "WEAK/INCONSISTENT"
    return "NO EFFECT"
