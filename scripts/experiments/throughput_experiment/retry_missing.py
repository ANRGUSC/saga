"""Purge and recompute results tainted by the GreedyInsert min_start_time bug.

GreedyInsert.call() used to search for an insertion gap as if min_start_time
were always 0, then clamp the result up to the real min_start_time afterward
-- without rechecking that the shifted interval still fit the gap. Any
scheduler that drives an online environment with advancing simulated time
(passed in as min_start_time) could silently produce an invalid -- or merely
suboptimal but unflagged -- placement. That's been fixed in
saga/schedulers/parametric/components.py (current_moment is now passed into
the gap search itself), but every existing row produced through that path is
suspect and needs to be recomputed, not just the ones that errored out.

Affected scheduler columns (every dataset, including ones that look "complete"):
    FIFO                  -- regular scheduler; runs FIFOEnvironment internally
    Inspirit_FIFO_*_*     -- 9 sweep combos
    Inspirit_HEFT_*_*     -- 9 sweep combos
    Inspirit_CPoP_*_*     -- 9 sweep combos

Everything else (HEFT, CPoP, HEFT_Throughput, CPoP_Throughput, MaxMin, MinMin,
MCT, MET, MST, OLB, Sufferage, WBA, BIL, Duplex, ETF, FastestNode, FCP, FLB,
GDL, Hbmct, Msbc, Mt_Scheduler, Multi_Obj) is built from scratch with
min_start_time=0 in this experiment and never drives online time-stepping, so
those rows are unaffected and left alone.

Usage:
    python3 retry_missing.py [--dry-run]
"""
import argparse
import pathlib
import re
import shutil

import pandas as pd

from run_throughput import evaluate_dataset, resultsdir

_TAINTED_RE = re.compile(r"^(FIFO|Inspirit_(?:HEFT|CPoP|FIFO)_\d+_\d+)$")


def purge_tainted(path: pathlib.Path) -> int:
    df = pd.read_csv(path)
    tainted_mask = df["Scheduler"].str.match(_TAINTED_RE, na=False)
    removed = int(tainted_mask.sum())
    if removed:
        df = df[~tainted_mask]
        df.to_csv(path, index=False)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = sorted(resultsdir.glob("*.csv"))
    dataset_names = []
    for path in paths:
        if not args.dry_run:
            shutil.copy2(path, path.with_suffix(".csv.bak"))
        removed = purge_tainted(path) if not args.dry_run else int(
            pd.read_csv(path)["Scheduler"].str.match(_TAINTED_RE, na=False).sum()
        )
        if removed:
            print(f"{path.stem}: purged {removed} tainted rows")
            dataset_names.append(path.stem)

    print(f"\n{len(dataset_names)} datasets need recomputation: {dataset_names}")
    if args.dry_run:
        return

    for dataset_name in dataset_names:
        print(f"\nRecomputing tainted schedulers for {dataset_name}...")
        evaluate_dataset(dataset_name)


if __name__ == "__main__":
    main()
