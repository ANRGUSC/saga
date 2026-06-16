import os
import sys
import time
import pathlib

thisdir = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(thisdir))
sys.path.insert(0, str(thisdir.parent.parent.parent / "src"))
os.environ["SAGA_DATA_DIR"] = str(thisdir / "data")

from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import UpwardRanking, GreedyInsert, GreedyInsertCompareFuncs
from saga.schedulers.data import Dataset
from saga.schedulers.online import InspiritController, TaskCompletionStep, ReadyChangeObserver
from saga.schedulers.online.environment import Environment


if __name__ == "__main__":
    ds = Dataset(name="genome")
    inst = ds.get_instance(ds.instances[0])
    n_tasks = len(list(inst.task_graph.tasks))
    n_machines = len(list(inst.network.nodes))
    threshold, delta_ready = n_machines, n_machines
    print(f"Instance: {ds.instances[0]}  tasks={n_tasks}  machines={n_machines}", flush=True)
    print(f"threshold={threshold}  delta_ready={delta_ready}\n", flush=True)

    t0 = time.time()
    step_times = []

    def on_step(env):
        step_t = time.time() - t0
        step_times.append(step_t)
        rec = env.history[-1]
        heft_fired = env.controller.last_dispatched is not None
        print(
            f"  step={rec.step:4d}  t={rec.time:8.2f}  "
            f"done={len(rec.finished_tasks):3d}  running={len(rec.running_tasks):2d}  "
            f"ready={len(rec.ready_tasks):3d}  unready={len(rec.unready_tasks):3d}  "
            f"wall={step_t:.2f}s  heft={'YES' if heft_fired else 'no'}",
            flush=True,
        )

    env = Environment(
        network=inst.network,
        task_graph=inst.task_graph,
        scheduler=ParametricScheduler(
            initial_priority=UpwardRanking(),
            insert_task=GreedyInsert(
                append_only=False,
                compare=GreedyInsertCompareFuncs.EFT,
                critical_path=False,
            )
        ),
        step_strategy=TaskCompletionStep(),
        observer=ReadyChangeObserver(delta_ready),
        controller=InspiritController(
            dec_step=threshold,
            s_inc=threshold,
            s_dec=threshold,
        ),
        on_step=on_step,
    )
    schedule = env.run()
    total = time.time() - t0
    print(f"\nDone in {total:.1f}s  makespan={schedule.makespan:.2f}  throughput={schedule.throughput:.4f}", flush=True)
