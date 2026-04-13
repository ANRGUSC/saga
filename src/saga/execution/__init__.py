"""Run real task graphs, scheduled by SAGA, on real compute platforms.

This sub-package bridges three things:

* A tiny Python-decorator notation (``@saga_task``) for defining executable
  task graphs — similar in spirit to Dask / Parsl, but deliberately minimal
  and oriented towards SAGA's "related machines" scheduling model.
* SAGA's core types (``TaskGraph``, ``Network``, ``Schedule``).
* CCTools Makeflow / Work Queue as the actual execution backend on
  heterogeneous clusters such as Chameleon Cloud.

Typical workflow::

    from saga.execution import saga_task, SagaGraph
    from saga.execution.makeflow import emit_makeflow
    from saga.execution.workqueue import WorkQueueRunner
    from saga.schedulers import HeftScheduler
    from saga import Network

    @saga_task(inputs=[], outputs=["x"], cost=lambda cfg: 1.0)
    def seed(cfg):
        import numpy as np
        return {"x": np.arange(cfg["N"])}

    @saga_task(inputs=["x"], outputs=["y"],
               cost=lambda cfg: 10.0 * cfg["N"])
    def double(x, cfg):
        return {"y": x * 2}

    graph = SagaGraph([seed, double], config={"N": 10_000})
    task_graph = graph.to_task_graph()

    network = Network.create(
        nodes=[("head", 2.0e9), ("worker-1", 1.0e9)],
        edges=[("head", "worker-1", 1.25e8)],
    )
    schedule = HeftScheduler().schedule(network, task_graph)

    # Option A: emit Makeflow JSON for portable / local execution.
    artifacts = emit_makeflow(graph, schedule, "out/", task_module=__name__)

    # Option B: execute directly on a Work Queue cluster with per-node pinning.
    runner = WorkQueueRunner(graph, schedule, task_module=__name__)
    result = runner.run()

``dagprofiler`` integration: if you've profiled your workflow with
``dagprofiler``, call ``saga.execution.profile_adapter.profile_to_task_graph``
to load its JSON and feed the result to any SAGA scheduler.
"""

from .graph import SagaGraph, SagaTask, saga_task
from .profile_adapter import profile_to_task_graph

__all__ = [
    "SagaGraph",
    "SagaTask",
    "saga_task",
    "profile_to_task_graph",
]
