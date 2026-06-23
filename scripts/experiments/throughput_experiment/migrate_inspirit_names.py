"""One-time migration for results collected before the Inspirit sweep naming fix.

Old naming encoded the raw threshold/delta_ready values, which are n * multiplier
(n = num_processors for that instance). That meant a name like "Inspirit_HEFT_1_1"
only ever referred to single-processor instances, since threshold=1 requires n=1.
This rewrites Scheduler names to use the multiplier instead, by cross-referencing
each row's Instance against the saved dataset to recover n. No rescheduling needed.

Usage:
    python migrate_inspirit_names.py [--dry-run]
"""
import argparse
import pathlib
import re
import shutil
from typing import Dict, Tuple

import pandas as pd

from common import datadir  # noqa: F401  (sets SAGA_DATA_DIR as a side effect)
from saga.schedulers.data import Dataset

thisdir = pathlib.Path(__file__).parent.resolve()
resultsdir = thisdir / "results" / "throughput"

_OLD_INSPIRIT_RE = re.compile(r"^(Inspirit_(?:HEFT|CPoP|FIFO))_(\d+)_(\d+)$")


def _num_nodes(dataset_cache: Dict[str, Dataset], dataset_name: str, instance_name: str) -> int:
    if dataset_name not in dataset_cache:
        dataset_cache[dataset_name] = Dataset(name=dataset_name)
    instance = dataset_cache[dataset_name].get_instance(instance_name)
    return len(list(instance.network.nodes))


def migrate_csv(path: pathlib.Path) -> Tuple[pd.DataFrame, int]:
    df = pd.read_csv(path)
    dataset_cache: Dict[str, Dataset] = {}
    n_cache: Dict[Tuple[str, str], int] = {}
    renamed = 0
    new_names = []

    for _, row in df.iterrows():
        match = _OLD_INSPIRIT_RE.match(row["Scheduler"])
        if not match:
            new_names.append(row["Scheduler"])
            continue

        base, threshold, delta_ready = match.group(1), int(match.group(2)), int(match.group(3))
        key = (row["Dataset"], row["Instance"])
        if key not in n_cache:
            n_cache[key] = _num_nodes(dataset_cache, row["Dataset"], row["Instance"])
        n = n_cache[key]

        if threshold % n != 0 or delta_ready % n != 0:
            raise ValueError(
                f"{row['Dataset']}/{row['Instance']}: threshold={threshold}, "
                f"delta_ready={delta_ready} not a multiple of n={n}; "
                "can't infer multiplier."
            )
        t_mult, d_mult = threshold // n, delta_ready // n
        new_names.append(f"{base}_{t_mult}_{d_mult}")
        renamed += 1

    df["Scheduler"] = new_names
    return df, renamed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path in sorted(resultsdir.glob("*.csv")):
        df, renamed = migrate_csv(path)
        if renamed == 0:
            continue
        print(f"{path.name}: renamed {renamed} rows")
        if not args.dry_run:
            shutil.copy2(path, path.with_suffix(".csv.bak"))
            df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
