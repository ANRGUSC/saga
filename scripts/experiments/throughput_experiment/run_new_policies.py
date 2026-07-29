"""Backfill the new checkpoint reschedule policies into the existing stochastic
results CSVs, without recomputing the schedulers already in results/<branch>_stochastic.csv
(see run.py for the full scheduler grid).

Stochastic-only: reschedule policies collapse to static in the deterministic regime (see
run.py's config_names), so there is nothing to add to the deterministic CSVs.

checkpoint_10 is skipped for the riotbench branch: those dataflows usually have fewer than
10 tasks, so its 10%-interval checkpoints collapse into rescheduling on every step.

Usage:
    python run_new_policies.py riotbench
    python run_new_policies.py wfcommons [n_instances] [n_seeds]
"""
import logging
import sys
from multiprocessing import Pool

import filelock
import pandas as pd
from tqdm import tqdm

from common import resultsdir, num_processors
from instances import base_instances, scaled, workflows_for
from run import BASES, build_config, evaluate, CCRS, SEED

logging.basicConfig(level=logging.WARNING)

NEW_POLICIES = ["checkpoint_quarterly", "checkpoint_mid", "checkpoint_10"]


def new_config_names(branch: str = None):
    """New config names to backfill for a branch.

    checkpoint_10 is skipped for riotbench: those dataflows usually have fewer than 10
    tasks, so its 10%-interval checkpoints collapse into rescheduling on every step.
    """
    policies = NEW_POLICIES
    if branch == "riotbench":
        policies = [p for p in policies if p != "checkpoint_10"]
    return [f"{b}_{p}" for b in BASES for p in policies]


def _eval_instance(job):
    """Worker: run all seeds x new configs for one instance; return a list of result rows."""
    branch, workflow, ccr, instance, n_seeds = job
    rows = []
    for seed in range(n_seeds):
        for name in new_config_names(branch):
            scheduler, policy = build_config(name)
            try:
                result = evaluate("stochastic", scheduler, policy, instance, seed)
            except Exception as e:  # noqa: BLE001
                logging.warning("failed %s ccr=%s %s seed=%d: %s",
                                name, ccr, instance.name, seed, e)
                continue
            rows.append({
                "Branch": None, "Regime": "stochastic", "Workflow": workflow,
                "CCR": ccr, "Instance": instance.name, "Seed": seed,
                "Scheduler": name, "Throughput": result["Throughput"],
                "Makespan": result["Makespan"], "RescheduleCount": result["RescheduleCount"],
            })
    return rows


def run(branch: str, n_instances: int, n_seeds: int, workers: int) -> None:
    out = resultsdir / f"{branch}_stochastic.csv"
    lock_path = out.with_suffix(".csv.lock")

    # Resume: skip any (Workflow, CCR, Instance) that already has every seed x new-config
    # row, so a crash or a second invocation of this backfill doesn't duplicate rows.
    new_names = new_config_names(branch)
    expected_per_instance = n_seeds * len(new_names)
    finished_keys: set = set()
    if out.exists():
        prev = pd.read_csv(out)
        prev_new = prev[prev["Scheduler"].isin(new_names)]
        counts = prev_new.groupby(["Workflow", "CCR", "Instance"]).size()
        finished_keys = {k for k, n in counts.items() if n >= expected_per_instance}
        logging.warning("resuming: %d instances already have the new policies in %s",
                        len(finished_keys), out.name)

    jobs = []
    for workflow in workflows_for(branch):
        base = base_instances(branch, workflow, n_instances, "stochastic", seed=SEED)
        for ccr in CCRS:
            for instance in (scaled(b, ccr) for b in base):
                if (workflow, ccr, instance.name) in finished_keys:
                    continue
                jobs.append((branch, workflow, ccr, instance, n_seeds))

    if not jobs:
        print(f"nothing to do; {out} already has the new policies")
        return

    total_written = 0
    with Pool(workers) as pool:
        for result in tqdm(pool.imap_unordered(_eval_instance, jobs),
                           total=len(jobs), desc=f"{branch}/stochastic (new policies)", file=sys.stderr):
            for row in result:
                row["Branch"] = branch
            # Append under a filelock, matching run.py's crash-safety story.
            with filelock.FileLock(lock_path):
                df = pd.DataFrame(result)
                df.to_csv(out, mode="a", header=False, index=False)
            total_written += len(result)
    print(f"wrote {total_written} rows -> {out}")


def main() -> None:
    branch = sys.argv[1] if len(sys.argv) > 1 else "riotbench"
    n_instances = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    run(branch, n_instances, n_seeds, num_processors)


if __name__ == "__main__":
    main()
