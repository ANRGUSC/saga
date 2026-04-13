"""End-to-end example: define a task graph, schedule it with SAGA, emit
Makeflow JSON, and (optionally) execute it on Chameleon Cloud via Work Queue.

Run locally (no cluster required)::

    # From the saga/ root so that the 'tasks' module is importable:
    cd scripts/examples/dagprofiler_chameleon
    python main.py --mode emit

Run on Work Queue (with workers already up, see provision_workers.sh)::

    python main.py --mode wq --project saga-demo

The task definitions live in the sibling ``tasks.py`` so that Work Queue
workers — which invoke ``python -m saga.execution._wrapper --task-module
tasks …`` — can ``import tasks`` and find the decorated functions.  The
driver and the workers therefore share a single source of truth.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Make sure the sibling ``tasks`` module is importable whether you run from
# the example directory or from the project root.
_THIS_DIR = pathlib.Path(__file__).parent.resolve()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import tasks as user_tasks  # noqa: E402  (deliberate: path manipulation above)

from saga.execution import SagaGraph  # noqa: E402
from saga.execution.chameleon import ChameleonNode, build_network  # noqa: E402
from saga.execution.makeflow import emit_makeflow  # noqa: E402
from saga.schedulers import HeftScheduler  # noqa: E402


def build_graph(n: int, seed: int = 42) -> SagaGraph:
    """Assemble the four user tasks into a SagaGraph."""
    return SagaGraph(
        tasks=[
            user_tasks.generate,
            user_tasks.square,
            user_tasks.shift,
            user_tasks.combine,
        ],
        config={"N": n, "seed": seed},
    )


def build_demo_network():
    """A heterogeneous 3-node reservation — replace with your actual lease."""
    return build_network(
        [
            ChameleonNode("saga-worker-0", speed=2.0),
            ChameleonNode("saga-worker-1", speed=1.5),
            ChameleonNode("saga-worker-2", speed=1.0),
        ],
        default_bandwidth=100.0,  # arbitrary units — same scale as task costs
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("emit", "wq"),
        default="emit",
        help="'emit' writes a Makeflow JSON; 'wq' runs on Work Queue.",
    )
    ap.add_argument(
        "--out",
        default="./saga_chameleon_out",
        help="Output directory for the Makeflow JSON + staging files.",
    )
    ap.add_argument("--n", type=int, default=10_000, help="Per-task problem size N.")
    ap.add_argument(
        "--project", default=None, help="Work Queue project name (for --mode wq)."
    )
    ap.add_argument(
        "--port",
        type=int,
        default=9123,
        help="Work Queue manager port (for --mode wq).",
    )
    args = ap.parse_args(argv)

    graph = build_graph(n=args.n)
    network = build_demo_network()
    task_graph = graph.to_task_graph()
    schedule = HeftScheduler().schedule(network, task_graph)

    print(f"SAGA schedule makespan = {schedule.makespan:.2f}")
    for node_name, scheduled in schedule.mapping.items():
        if scheduled:
            print(f"  {node_name}: {[t.name for t in scheduled]}")

    if args.mode == "emit":
        artifacts = emit_makeflow(
            graph,
            schedule,
            args.out,
            task_module="tasks",
        )
        print(f"\nWrote Makeflow JSON to {artifacts.makeflow_json}")
        print("Local run:")
        print(f"  cd {pathlib.Path(artifacts.root).resolve()}")
        print(f"  makeflow --jx {artifacts.makeflow_json.name} -T local")
        print("Work Queue run:")
        print(
            f"  makeflow --jx {artifacts.makeflow_json.name} "
            f"-T wq -N {args.project or 'saga-demo'}"
        )
        return 0

    # args.mode == "wq"
    from saga.execution.workqueue import WorkQueueRunner

    runner = WorkQueueRunner(
        graph,
        schedule,
        task_module="tasks",
        port=args.port,
        project_name=args.project,
        staging_dir=pathlib.Path(args.out),
    )
    print("Starting Work Queue manager.  Launch workers with:")
    print(
        f"  provision_workers.sh {args.project or 'saga-demo'} "
        "saga-worker-0 saga-worker-1 saga-worker-2"
    )
    result = runner.run(log_callback=print)
    print(f"\nDone.  Observed makespan: {result.makespan_seconds:.2f}s")
    print(f"Per-task seconds: {result.per_task_seconds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
