"""
Compare FrontierHEFT vs FIFO on a Montage scientific workflow instance.

Both schedulers print per-step diagnostics via on_step(), then a final
summary is printed showing makespan and throughput side-by-side.
"""

import pathlib
import sys
import time

thisdir = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(thisdir.parent.parent.parent / "src"))

from saga.schedulers.data.wfcommons import get_networks, get_workflows
from saga.schedulers.online.online_algorithms.frontier_heft import FrontierHeftEnvironment
from saga.schedulers.online.online_algorithms.FIFO import FIFOEnvironment


# ---------------------------------------------------------------------------
# on_step callbacks
# ---------------------------------------------------------------------------

def make_on_step(label: str, t0: float, total_tasks: int):
    """Return an on_step callback that prints per-step diagnostics.

    Uses env.frontier_set directly because StepRecord.ready_tasks only captures
    tasks *newly* added to the frontier this step, and unready_tasks is never
    populated by FrontierEnvironment (it starts empty and is only discarded from).
    """
    def on_step(env) -> None:
        rec = env.history[-1]
        wall = time.time() - t0
        n_done = len(rec.finished_tasks)
        n_running = len(rec.running_tasks)
        n_committed = sum(len(tasks) for tasks in env.schedule.mapping.values())
        n_frontier = len(env.frontier_set)
        n_unready = total_tasks - n_committed - n_frontier
        print(
            f"  [{label}] step={rec.step:>4}  sim_t={rec.time:>12.4f}"
            f"  done={n_done:>4}  running={n_running:>3}"
            f"  frontier={n_frontier:>4}  unready={n_unready:>4}"
            f"  makespan={rec.makespan:>12.4f}  wall={wall:.2f}s",
            flush=True,
        )
    return on_step


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_frontier_heft(network, task_graph):
    t0 = time.time()
    n_tasks = len(list(task_graph.tasks))
    print(f"\n{'='*70}")
    print("  FrontierHEFT")
    print(f"{'='*70}")
    env = FrontierHeftEnvironment(network, task_graph, on_step=make_on_step("FrontierHEFT", t0, n_tasks))
    schedule = env.run()
    wall = time.time() - t0
    print(
        f"\n  [FrontierHEFT] DONE  steps={env._step}"
        f"  makespan={schedule.makespan:.4f}"
        f"  throughput={schedule.throughput:.6f}"
        f"  wall={wall:.2f}s",
        flush=True,
    )
    return schedule


def run_fifo(network, task_graph):
    t0 = time.time()
    n_tasks = len(list(task_graph.tasks))
    print(f"\n{'='*70}")
    print("  FIFO")
    print(f"{'='*70}")
    env = FIFOEnvironment(network, task_graph, on_step=make_on_step("FIFO", t0, n_tasks))
    schedule = env.run()
    wall = time.time() - t0
    print(
        f"\n  [FIFO] DONE  steps={env._step}"
        f"  makespan={schedule.makespan:.4f}"
        f"  throughput={schedule.throughput:.6f}"
        f"  wall={wall:.2f}s",
        flush=True,
    )
    return schedule


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating Montage workflow instance...")
    networks = get_networks(num=1, cloud_name="chameleon")
    workflows = get_workflows(num=1, recipe_name="montage")
    network = networks[0]
    task_graph = workflows[0]

    n_tasks = len(list(task_graph.tasks))
    n_nodes = len(list(network.nodes))
    print(f"Instance: {n_tasks} tasks, {n_nodes} compute nodes\n")

    heft_schedule = run_frontier_heft(network, task_graph)
    fifo_schedule = run_fifo(network, task_graph)

    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    print(f"  {'Scheduler':<20}  {'Makespan':>12}  {'Throughput':>12}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}")
    print(f"  {'FrontierHEFT':<20}  {heft_schedule.makespan:>12.4f}  {heft_schedule.throughput:>12.6f}")
    print(f"  {'FIFO':<20}  {fifo_schedule.makespan:>12.4f}  {fifo_schedule.throughput:>12.6f}")

    heft_ratio = heft_schedule.makespan / fifo_schedule.makespan if fifo_schedule.makespan > 0 else float("nan")
    winner = "FrontierHEFT" if heft_schedule.makespan < fifo_schedule.makespan else "FIFO"
    print(f"\n  Winner (makespan): {winner}")
    print(f"  Makespan ratio (HEFT/FIFO): {heft_ratio:.4f}")


if __name__ == "__main__":
    main()
