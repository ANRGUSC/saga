"""
Sequential montage debugger.

Iterates over every montage instance one at a time (no multiprocessing) so that
the failing instance and scheduler combination can be identified precisely.

For each instance it runs:
  - all regular schedulers
  - all sweepable (Inspirit) scheduler combinations

Every scheduler run logs per-step diagnostics via the on_step callback.
Exceptions are caught, printed with a full traceback, and the loop continues.
"""

import os
import pathlib
import sys
import time
import traceback

thisdir = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(thisdir))
sys.path.insert(0, str(thisdir.parent.parent.parent / "src"))
os.environ["SAGA_DATA_DIR"] = str(thisdir / "data")

import saga.schedulers as s
from saga.schedulers.data import Dataset
from saga.schedulers.throughput.inspirit import InspriritScheduler
from saga.schedulers.throughput.mt_scheduler import MTScheduler
from saga.schedulers.throughput.multi_obj import MultiObjScheduler
from saga.schedulers import HeftScheduler, CpopScheduler
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    UpwardRanking, CPoPRanking, GreedyInsert, GreedyInsertCompareFuncs,
)
from saga.schedulers.online import (
    InspiritController, InspiritEnvironment, TaskCompletionStep, ReadyChangeObserver,
)
from saga.schedulers.online.online_algorithms.FIFO import FIFOScheduler, InspiritFIFOScheduler


# ---------------------------------------------------------------------------
# Scheduler catalogue (mirrors run_throughput.py)
# ---------------------------------------------------------------------------

REGULAR_SCHEDULERS = {
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

SWEEPABLE_FACTORIES = {
    "Inspirit_HEFT": lambda t, d: InspriritScheduler(
        ParametricScheduler(
            initial_priority=UpwardRanking(),
            insert_task=GreedyInsert(
                append_only=False,
                compare=GreedyInsertCompareFuncs.EFT,
                critical_path=False,
            ),
        ),
        t, d,
    ),
    "Inspirit_CPoP": lambda t, d: InspriritScheduler(
        ParametricScheduler(
            initial_priority=CPoPRanking(),
            insert_task=GreedyInsert(
                append_only=False,
                compare=GreedyInsertCompareFuncs.EFT,
                critical_path=True,
            ),
        ),
        t, d,
    ),
    "Inspirit_FIFO": lambda t, d: InspiritFIFOScheduler(t, d),
}


# ---------------------------------------------------------------------------
# Inspirit-aware schedule runner with per-step logging
# ---------------------------------------------------------------------------

def _make_on_step(label: str, t0: float):
    def on_step(env):
        elapsed = time.time() - t0
        rec = env.history[-1]
        print(
            f"  [{label}] step={rec.step:4d}  sim_t={rec.time:10.3f}"
            f"  done={len(rec.finished_tasks):3d}  running={len(rec.running_tasks):2d}"
            f"  ready={len(rec.ready_tasks):3d}  unready={len(rec.unready_tasks):3d}"
            f"  makespan={rec.makespan:10.3f}  wall={elapsed:.2f}s",
            flush=True,
        )
    return on_step


def _run_inspirit(label: str, network, task_graph, threshold: int, delta_ready: int,
                  base_scheduler, smoothing_rate: float = 0.8):
    """Run one Inspirit variant with per-step debug output; return the schedule."""
    t0 = time.time()
    env = InspiritEnvironment(
        network=network,
        task_graph=task_graph,
        scheduler=base_scheduler,
        step_strategy=TaskCompletionStep(),
        observer=ReadyChangeObserver(delta_ready),
        time_window=None,
        controller=InspiritController(smoothing_rate=smoothing_rate),
        on_step=_make_on_step(label, t0),
        dec_step=threshold,
        s_inc=threshold,
        s_dec=threshold,
    )
    schedule = env.run()
    elapsed = time.time() - t0
    print(
        f"  [{label}] DONE  makespan={schedule.makespan:.3f}"
        f"  throughput={schedule.throughput:.4f}  wall={elapsed:.2f}s",
        flush=True,
    )
    return schedule


def _run_regular(label: str, scheduler, network, task_graph):
    """Run a regular (non-Inspirit) scheduler; return the schedule."""
    t0 = time.time()
    schedule = scheduler.schedule(network=network, task_graph=task_graph)
    elapsed = time.time() - t0
    print(
        f"  [{label}] DONE  makespan={schedule.makespan:.3f}"
        f"  throughput={schedule.throughput:.8f}  wall={elapsed:.2f}s",
        flush=True,
    )
    return schedule


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    dataset = Dataset(name="montage")
    instances = dataset.instances
    n_instances = len(instances)
    print(f"=== Montage dataset: {n_instances} instances ===\n", flush=True)

    for idx, inst_name in enumerate(instances):
        print(
            f"\n{'='*70}\n"
            f"[{idx+1}/{n_instances}] Instance: {inst_name}\n"
            f"{'='*70}",
            flush=True,
        )

        try:
            instance = dataset.get_instance(inst_name)
        except Exception:
            print(f"  ERROR loading instance {inst_name}:", flush=True)
            traceback.print_exc()
            continue

        n_tasks = len(list(instance.task_graph.tasks))
        n_machines = len(list(instance.network.nodes))
        print(f"  tasks={n_tasks}  machines={n_machines}", flush=True)

        # --- Regular schedulers ---
        for sched_name, scheduler in REGULAR_SCHEDULERS.items():
            print(f"\n  >> {sched_name}", flush=True)
            try:
                _run_regular(sched_name, scheduler, instance.network, instance.task_graph)
            except Exception:
                print(f"  EXCEPTION in {sched_name} on {inst_name}:", flush=True)
                traceback.print_exc()

        # --- Sweepable (Inspirit) schedulers ---
        n = n_machines
        thresholds = sorted({max(1, n), n * 2, n * 3})
        delta_readys = sorted({max(1, n), n * 2, n * 3})

        for base_name, factory in SWEEPABLE_FACTORIES.items():
            for threshold in thresholds:
                for delta_ready in delta_readys:
                    sched_name = f"{base_name}_{threshold}_{delta_ready}"
                    print(f"\n  >> {sched_name}  threshold={threshold}  delta_ready={delta_ready}", flush=True)

                    # Build the underlying base scheduler fresh each time (not reusable across runs)
                    if base_name == "Inspirit_HEFT":
                        base_sched = ParametricScheduler(
                            initial_priority=UpwardRanking(),
                            insert_task=GreedyInsert(
                                append_only=False,
                                compare=GreedyInsertCompareFuncs.EFT,
                                critical_path=False,
                            ),
                        )
                        try:
                            _run_inspirit(
                                sched_name, instance.network, instance.task_graph,
                                threshold, delta_ready, base_sched,
                            )
                        except Exception:
                            print(f"  EXCEPTION in {sched_name} on {inst_name}:", flush=True)
                            traceback.print_exc()

                    elif base_name == "Inspirit_CPoP":
                        base_sched = ParametricScheduler(
                            initial_priority=CPoPRanking(),
                            insert_task=GreedyInsert(
                                append_only=False,
                                compare=GreedyInsertCompareFuncs.EFT,
                                critical_path=True,
                            ),
                        )
                        try:
                            _run_inspirit(
                                sched_name, instance.network, instance.task_graph,
                                threshold, delta_ready, base_sched,
                            )
                        except Exception:
                            print(f"  EXCEPTION in {sched_name} on {inst_name}:", flush=True)
                            traceback.print_exc()

                    elif base_name == "Inspirit_FIFO":
                        # InspiritFIFOScheduler calls env.run() internally — wrap it to get step logging
                        try:
                            sched = InspiritFIFOScheduler(threshold, delta_ready)
                            t0 = time.time()
                            schedule = sched.schedule(
                                network=instance.network, task_graph=instance.task_graph
                            )
                            elapsed = time.time() - t0
                            print(
                                f"  [{sched_name}] DONE  makespan={schedule.makespan:.3f}"
                                f"  throughput={schedule.throughput:.8f}  wall={elapsed:.2f}s",
                                flush=True,
                            )
                        except Exception:
                            print(f"  EXCEPTION in {sched_name} on {inst_name}:", flush=True)
                            traceback.print_exc()


if __name__ == "__main__":
    main()
