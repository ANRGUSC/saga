"""Throughput experiment driver (see EXPERIMENT_PLAN.md).

Runs one branch x regime over the scheduler grid and writes a tidy CSV of realized
throughput, one row per (workflow, ccr, instance, seed, scheduler).

Schedulers: the four bases (HEFT, CPoP, HEFT-Tp, CPoP-Tp) under each policy, plus
FastestNode. HEFT/CPoP are the classic EFT configs; the -Tp variants swap the placement
comparator for the throughput bottleneck. Policies are static and inspirit in both
regimes, plus reschedule in the stochastic regime (offline it equals static).

Evaluation is parallelized across instances with a small process pool (capped at 4).

Usage:
    python run.py riotbench deterministic
    python run.py riotbench stochastic
    python run.py wfcommons deterministic [n_instances] [n_seeds]
    python run.py wfcommons stochastic   [n_instances] [n_seeds]
"""
import logging
import os
import sys
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from common import resultsdir
from instances import base_instances, scaled, workflows_for

from saga.schedulers import FastestNodeScheduler
from saga.schedulers.online.environment import Environment, StochasticEnvironment
from saga.schedulers.online.policy import ReschedulePolicy, InspiritPolicy
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    UpwardRanking, CPoPRanking, GreedyInsert, GreedyInsertCompareFuncs,
)

logging.basicConfig(level=logging.WARNING)

CCRS = [0.2, 0.5, 1.0, 2.0, 5.0]
SEED = 0
MAX_WORKERS = 4  # cap the pool regardless of core count

EFT = GreedyInsertCompareFuncs.EFT
TP = GreedyInsertCompareFuncs.Throughput


def _mean(rv):
    return rv.mean()


def _base(priority_cls, critical_path, compare):
    return ParametricScheduler(
        initial_priority=priority_cls(),
        insert_task=GreedyInsert(compare=compare, critical_path=critical_path),
    )


# base name -> factory (fresh scheduler per use; the -Tp variants only swap the comparator)
BASES = {
    "HEFT": lambda: _base(UpwardRanking, False, EFT),
    "CPoP": lambda: _base(CPoPRanking, True, EFT),
    "HEFT-Tp": lambda: _base(UpwardRanking, False, TP),
    "CPoP-Tp": lambda: _base(CPoPRanking, True, TP),
}
_POLICIES = {"static": lambda: None, "inspirit": InspiritPolicy, "reschedule": ReschedulePolicy}


def config_names(regime: str):
    """Config names in the scheduler grid for a regime (reschedule only when stochastic)."""
    policies = ["static", "inspirit"] + (["reschedule"] if regime == "stochastic" else [])
    return [f"{b}_{p}" for b in BASES for p in policies] + ["FastestNode"]


def build_config(name: str):
    """Reconstruct (scheduler, policy) from a config name (picklable across processes)."""
    if name == "FastestNode":
        return FastestNodeScheduler(), None
    base, policy = name.split("_", 1)
    return BASES[base](), _POLICIES[policy]()


def evaluate(regime, scheduler, policy, instance, seed) -> float:
    """Return the realized throughput of one config on one instance/seed."""
    cons = instance.node_constraints
    if regime == "deterministic":
        if policy is None:
            schedule = scheduler.schedule(instance.network, instance.task_graph, node_constraints=cons)
        else:
            schedule = Environment(
                instance.network, instance.task_graph,
                scheduler=scheduler, policy=policy, node_constraints=cons,
            ).run()
    else:
        schedule = StochasticEnvironment(
            instance.network, instance.task_graph,
            scheduler=scheduler, estimate=_mean, policy=policy,
            seed=seed, node_constraints=cons,
        ).run()
    return schedule.throughput


def _eval_instance(job):
    """Worker: run all seeds x configs for one instance; return a list of result rows."""
    regime, workflow, ccr, instance, n_seeds = job
    seeds = range(n_seeds) if regime == "stochastic" else [0]
    rows = []
    for seed in seeds:
        for name in config_names(regime):
            scheduler, policy = build_config(name)
            try:
                throughput = evaluate(regime, scheduler, policy, instance, seed)
            except Exception as e:  # noqa: BLE001
                logging.warning("failed %s/%s ccr=%s %s seed=%d: %s",
                                workflow, name, ccr, instance.name, seed, e)
                continue
            rows.append({
                "Branch": None, "Regime": regime, "Workflow": workflow,
                "CCR": ccr, "Instance": instance.name, "Seed": seed,
                "Scheduler": name, "Throughput": throughput,
            })
    return rows


def run(branch: str, regime: str, n_instances: int, n_seeds: int, workers: int) -> None:
    resultsdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for workflow in workflows_for(branch):
        base = base_instances(branch, workflow, n_instances, regime, seed=SEED)
        for ccr in CCRS:
            for instance in (scaled(b, ccr) for b in base):
                jobs.append((regime, workflow, ccr, instance, n_seeds))

    rows = []
    with Pool(workers) as pool:
        for result in tqdm(pool.imap_unordered(_eval_instance, jobs),
                           total=len(jobs), desc=f"{branch}/{regime}", file=sys.stderr):
            rows.extend(result)

    for row in rows:
        row["Branch"] = branch
    out = resultsdir / f"{branch}_{regime}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


def main() -> None:
    branch = sys.argv[1] if len(sys.argv) > 1 else "riotbench"
    regime = sys.argv[2] if len(sys.argv) > 2 else "deterministic"
    default_inst, default_seeds = (10, 10) if regime == "stochastic" else (30, 1)
    n_instances = int(sys.argv[3]) if len(sys.argv) > 3 else default_inst
    n_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else default_seeds
    workers = min(MAX_WORKERS, os.cpu_count() or 1)
    run(branch, regime, n_instances, n_seeds, workers)


if __name__ == "__main__":
    main()
