import logging
import pathlib
import random
import sys
from multiprocessing import Pool
from typing import Callable, Dict, List, Tuple

import filelock
import pandas as pd
from tqdm import tqdm

import saga.schedulers as s
from common import datadir, num_processors
from saga import Scheduler
from saga.schedulers.data import Dataset
from saga.schedulers.throughput.mt_scheduler import MTScheduler
from saga.schedulers.throughput.multi_obj import MultiObjScheduler
from saga.schedulers.throughput.inspirit import InspriritScheduler
from saga.schedulers import HeftScheduler, CpopScheduler
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import UpwardRanking, CPoPRanking, GreedyInsert, GreedyInsertCompareFuncs
from saga.schedulers.online.online_algorithms.FIFO import FIFOScheduler, InspiritFIFOScheduler

logging.basicConfig(level=logging.ERROR)

thisdir = pathlib.Path(__file__).parent.resolve()
resultsdir = thisdir / "results" / "throughput"

# ---------------------------------------------------------------------------
# Regular schedulers — standard schedule(network, task_graph) interface
# ---------------------------------------------------------------------------

schedulers: Dict[str, Scheduler] = {
    "HEFT": HeftScheduler(),
    "CPoP": CpopScheduler(),
    "HEFT_Throughput": ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(
            append_only=False,
            compare=GreedyInsertCompareFuncs.Throughput,
            critical_path=False,
        ),
    ),
    "CPoP_Throughput": ParametricScheduler(
        initial_priority=CPoPRanking(),
        insert_task=GreedyInsert(
            append_only=False,
            compare=GreedyInsertCompareFuncs.Throughput,
            critical_path=True,
        ),
    ),
    "FIFO": FIFOScheduler(),
    "Mt_Scheduler": MTScheduler(),
    "Multi_Obj": MultiObjScheduler(),
    "BILScheduler": s.BILScheduler(),
    "DPSScheduler": s.DPSScheduler(),
    "DuplexScheduler": s.DuplexScheduler(),
    "ETFScheduler": s.ETFScheduler(),
    "FastestNodeScheduler": s.FastestNodeScheduler(),
    "FCPScheduler": s.FCPScheduler(),
    "FLBScheduler": s.FLBScheduler(),
    "GDLScheduler": s.GDLScheduler(),
    "HbmctScheduler": s.HbmctScheduler(),
    "HeftScheduler": s.HeftScheduler(),
    "MaxMinScheduler": s.MaxMinScheduler(),
    "MCTScheduler": s.MCTScheduler(),
    "METScheduler": s.METScheduler(),
    "MinMinScheduler": s.MinMinScheduler(),
    "MsbcScheduler": s.MsbcScheduler(),
    "MSTScheduler": s.MSTScheduler(),
    "OLBScheduler": s.OLBScheduler(),
    "SufferageScheduler": s.SufferageScheduler(),
    "WBAScheduler": s.WBAScheduler(),
}


# ---------------------------------------------------------------------------
# Sweepable scheduler factories
# Inspirit schedulers need (threshold, delta_ready) that scale with num_workers.
# Factory functions must be module-level (not lambdas) to be picklable by Pool.
# ---------------------------------------------------------------------------

def _make_inspirit_heft(threshold: int, delta_ready: int) -> Scheduler:
    return InspriritScheduler(
        ParametricScheduler(
            initial_priority=UpwardRanking(),
            insert_task=GreedyInsert(
                append_only=False,
                compare=GreedyInsertCompareFuncs.EFT,
                critical_path=False,
            )
        ), 
        threshold, 
        delta_ready)


def _make_inspirit_cpop(threshold: int, delta_ready: int) -> Scheduler:
    return InspriritScheduler(
        ParametricScheduler(
            initial_priority=CPoPRanking(),
            insert_task=GreedyInsert(
                append_only=False,
                compare=GreedyInsertCompareFuncs.EFT,
                critical_path=True,
            )
        ), 
        threshold, 
        delta_ready)


def _make_inspirit_fifo(threshold: int, delta_ready: int) -> Scheduler:
    return InspiritFIFOScheduler(threshold, delta_ready)


# Maps base name -> factory(threshold, delta_ready) -> Scheduler.
# Each combination becomes a separate row named "BaseName_threshold_deltaready".
sweepable_scheduler_factories: Dict[str, Callable[[int, int], Scheduler]] = {
    "Inspirit_HEFT": _make_inspirit_heft,
    "Inspirit_CPoP": _make_inspirit_cpop,
    "Inspirit_FIFO": _make_inspirit_fifo,
}


# ---------------------------------------------------------------------------
# Multiprocessing worker state
# ---------------------------------------------------------------------------

_worker_resultsdir: pathlib.Path
_worker_schedulers: Dict[str, Scheduler]
_worker_sweepable_factories: Dict[str, Callable[[int, int], Scheduler]]


def _init_worker(
    results_dir: pathlib.Path,
    sched: Dict[str, Scheduler],
    sweepable: Dict[str, Callable[[int, int], Scheduler]],
) -> None:
    global _worker_resultsdir, _worker_schedulers, _worker_sweepable_factories
    _worker_resultsdir = results_dir
    _worker_schedulers = sched
    _worker_sweepable_factories = sweepable


def _save_result(result: Dict, savepath: pathlib.Path, lock_path: pathlib.Path) -> None:
    with filelock.FileLock(lock_path):
        result_df = pd.DataFrame([result])
        if savepath.exists():
            result_df.to_csv(savepath, mode="a", header=False, index=False)
        else:
            result_df.to_csv(savepath, index=False)


def _already_done(
    dataset_name: str, instance_name: str, scheduler_name: str,
    savepath: pathlib.Path, lock_path: pathlib.Path,
) -> bool:
    with filelock.FileLock(lock_path):
        if savepath.exists():
            finished_df = pd.read_csv(savepath)
            finished = set(
                zip(finished_df["Dataset"], finished_df["Instance"], finished_df["Scheduler"])
            )
            return (dataset_name, instance_name, scheduler_name) in finished
    return False


def _evaluate_instance(args: Tuple[str, str]) -> List[Dict]:
    dataset_name, instance_name = args
    print(f"[{dataset_name}/{instance_name}] starting", file=sys.stderr, flush=True)
    dataset = Dataset(name=dataset_name)
    instance = dataset.get_instance(instance_name)
    savepath = _worker_resultsdir / f"{dataset_name}.csv"
    lock_path = savepath.with_suffix(".csv.lock")

    results = []

    # --- Regular schedulers ---
    for scheduler_name, scheduler in _worker_schedulers.items():
        if _already_done(dataset_name, instance_name, scheduler_name, savepath, lock_path):
            continue
        print(f"[{dataset_name}/{instance_name}] trying:   {scheduler_name}", flush=True)
        try:
            schedule = scheduler.schedule(network=instance.network, task_graph=instance.task_graph)
            result = {
                "Dataset": dataset_name,
                "Instance": instance_name,
                "Scheduler": scheduler_name,
                "Makespan": schedule.makespan,
                "Throughput": schedule.throughput,
            }
        except Exception as e:
            logging.warning("Failed %s/%s/%s: %s", dataset_name, instance_name, scheduler_name, e)
            continue
        print(f"[{dataset_name}/{instance_name}] finished: {scheduler_name}", flush=True)
        results.append(result)
        _save_result(result, savepath, lock_path)
    # --- Sweepable schedulers: expand over (threshold, delta_ready) ---
    n = len(list(instance.network.nodes))
    thresholds = sorted({max(1, n), n * 2, n * 3})
    delta_readys = sorted({max(1, n), n * 2, n * 3})

    for base_name, factory in _worker_sweepable_factories.items():
        for threshold in thresholds:
            for delta_ready in delta_readys:
                scheduler_name = f"{base_name}_{threshold}_{delta_ready}"
                if _already_done(dataset_name, instance_name, scheduler_name, savepath, lock_path):
                    continue
                print(f"[{dataset_name}/{instance_name}] trying:   {scheduler_name}", flush=True)
                try:
                    sched = factory(threshold, delta_ready)
                    schedule = sched.schedule(network=instance.network, task_graph=instance.task_graph)
                    result = {
                        "Dataset": dataset_name,
                        "Instance": instance_name,
                        "Scheduler": scheduler_name,
                        "Makespan": schedule.makespan,
                        "Throughput": schedule.throughput,
                    }
                except Exception as e:
                    logging.warning(
                        "Failed %s/%s/%s: %s", dataset_name, instance_name, scheduler_name, e
                    )
                    continue
                print(f"[{dataset_name}/{instance_name}] finished: {scheduler_name}", flush=True)
                results.append(result)
                _save_result(result, savepath, lock_path)

    return results


def evaluate_dataset(
    dataset_name: str,
    num_workers: int = num_processors,
    seed: int = 42,
) -> None:
    savepath = resultsdir / f"{dataset_name}.csv"
    savepath.parent.mkdir(exist_ok=True, parents=True)

    dataset = Dataset(name=dataset_name)
    work_items = [(dataset_name, name) for name in dataset.instances]
    random.Random(seed).shuffle(work_items)

    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(resultsdir, schedulers, sweepable_scheduler_factories),
    ) as pool:
        list(tqdm(
            pool.imap_unordered(_evaluate_instance, work_items),
            total=len(work_items),
            desc=f"Evaluating {dataset_name}",
            unit="instance",
            file=sys.stderr,
        ))


def main():
    datasets = [path.name for path in datadir.iterdir() if path.is_dir()]
    #removing already finished datasets
    for dataset in datasets:
        if dataset in {"epigenomics", "genome"}:
            continue
        evaluate_dataset(dataset)


if __name__ == "__main__":
    main()
