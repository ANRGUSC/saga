"""Execute a SAGA schedule on a CCTools Work Queue cluster.

This module does the one thing Makeflow's JSON schema can't express directly:
pin each task to the specific worker SAGA assigned.  It does so by driving
the Work Queue Python API (``work_queue`` or its successor ``ndcctools``)
directly:

* Every worker on the cluster is launched with a unique
  ``work_queue_worker --feature saga-node-<N>`` tag (see
  ``scripts/examples/dagprofiler_chameleon/provision_workers.sh``).
* For each task the manager submits a ``Task`` whose command is the SAGA
  wrapper (``python -m saga.execution._wrapper ...``) and on which
  ``Task.specify_feature("saga-node-<N>")`` has been called.
* Work Queue then *only* delivers that task to the matching worker.

Because ``work_queue`` is an optional, C-extension-backed dependency, this
file imports it lazily — the rest of SAGA keeps working even when CCTools is
not installed.
"""

from __future__ import annotations

import pathlib
import pickle
import shlex
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from saga import Schedule

from .graph import SagaGraph
from .makeflow import _category_name, _output_filename


def _lazy_import_work_queue():
    """Import CCTools' Work Queue bindings, with a helpful error message."""
    try:
        import work_queue as wq  # type: ignore[import-not-found]

        return wq
    except ImportError:  # pragma: no cover - requires cctools
        pass
    try:
        from ndcctools import work_queue as wq  # type: ignore[import-not-found]

        return wq
    except ImportError as exc:  # pragma: no cover - requires cctools
        raise RuntimeError(
            "Work Queue Python bindings are not installed.  On Chameleon / "
            "Ubuntu you can install CCTools with `conda install -c conda-forge "
            "ndcctools` (which includes the `work_queue` module)."
        ) from exc


def feature_for(node_name: str) -> str:
    """Canonical Work Queue feature string for a SAGA node."""
    safe = "".join(c if c.isalnum() else "_" for c in node_name)
    return f"saga-node-{safe}"


@dataclass
class WorkQueueRunResult:
    """Outcome of a Work Queue run."""

    makespan_seconds: float
    per_task_seconds: Dict[str, float] = field(default_factory=dict)
    output_files: Dict[str, pathlib.Path] = field(default_factory=dict)


class WorkQueueRunner:
    """Drive CCTools Work Queue to execute a SAGA-scheduled graph.

    Args:
        graph: The ``SagaGraph`` being executed.
        schedule: SAGA schedule assigning each task to a node.
        task_module: Dotted Python module that defines every task
            (e.g. ``my_pkg.pipeline``).  Workers must be able to
            ``import`` this module; the simplest way is to install the
            user's package on every worker node or to ship it as a
            tarball via ``extra_input_files``.
        task_module_map: Per-task override of ``task_module``.
        port: Work Queue manager port.  Use 0 to let Work Queue pick one.
        project_name: Optional project name for catalog-based worker
            discovery.  When set, workers can find the manager with
            ``work_queue_worker -N <project_name>`` instead of ``HOST:PORT``.
        python_bin: Python interpreter on the workers.
        extra_input_files: Additional files to attach as inputs of every
            task (e.g. a wheel / tarball of the user's package).
        staging_dir: Working directory for serialised config / inputs /
            outputs.  Defaults to a fresh tempdir.
        require_features: If ``True`` (default), each task specifies its
            target-node feature so Work Queue will only run it on the
            right worker.  Set to ``False`` for an unpinned dry run.
    """

    def __init__(
        self,
        graph: SagaGraph,
        schedule: Schedule,
        *,
        task_module: Optional[str] = None,
        task_module_map: Optional[Dict[str, str]] = None,
        port: int = 9123,
        project_name: Optional[str] = None,
        python_bin: str = "python3",
        extra_input_files: Optional[List[pathlib.Path]] = None,
        staging_dir: Optional[pathlib.Path] = None,
        require_features: bool = True,
    ):
        if task_module is None and not task_module_map:
            raise ValueError(
                "task_module= or task_module_map= is required so workers "
                "know which module to import"
            )
        self.graph = graph
        self.schedule = schedule
        self.task_module = task_module
        self.task_module_map = dict(task_module_map or {})
        self.port = port
        self.project_name = project_name
        self.python_bin = python_bin
        self.extra_input_files = [pathlib.Path(p) for p in (extra_input_files or [])]
        self.require_features = require_features
        self.staging_dir = pathlib.Path(
            staging_dir or tempfile.mkdtemp(prefix="saga-wq-")
        )
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        timeout_seconds: Optional[float] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> WorkQueueRunResult:
        """Submit every task, wait for completion, return timing info."""
        wq = _lazy_import_work_queue()

        log = log_callback or (lambda msg: None)

        manager = wq.WorkQueue(self.port)
        if self.project_name:
            manager.specify_name(self.project_name)
        log(f"Work Queue manager listening on port {manager.port}")

        # Pickle the config once; every task depends on it.
        config_path = self.staging_dir / "config.pkl"
        with config_path.open("wb") as fh:
            pickle.dump(dict(self.graph.config), fh, protocol=pickle.HIGHEST_PROTOCOL)

        # Map task name -> node name.
        task_to_node = self._task_to_node_map()

        wq_tasks: Dict[int, str] = {}  # wq task id -> SAGA task name
        submitted_order: List[str] = []

        for name, saga_task in self.graph.tasks.items():
            node = task_to_node[name]
            module = self.task_module_map.get(name, self.task_module)
            if module is None:
                raise ValueError(f"no module for task {name!r}")

            # Build the complete command string first (Work Queue's ``Task``
            # constructor takes it once and doesn't support amendment).
            cmd_parts: List[str] = [
                self.python_bin,
                "-m",
                "saga.execution._wrapper",
                "--task-module",
                module,
                "--task-attr",
                name,
                "--config",
                config_path.name,
            ]
            for inp in saga_task.inputs:
                producer = self.graph.producer_of(inp)
                if producer is None:
                    raise ValueError(f"task {name!r} consumes {inp!r} with no producer")
                fname = _output_filename(producer, inp)
                cmd_parts += ["--input", f"{inp}={fname}"]
            for out in saga_task.outputs:
                fname = _output_filename(name, out)
                cmd_parts += ["--output", f"{out}={fname}"]
            command = " ".join(shlex.quote(p) for p in cmd_parts)

            wqt = wq.Task(command)

            # The config pickle is staged once and cached across tasks.
            wqt.specify_input_file(str(config_path), config_path.name, cache=True)
            for extra in self.extra_input_files:
                wqt.specify_input_file(str(extra), extra.name, cache=True)

            # Stage input and output files corresponding to the dependencies.
            for inp in saga_task.inputs:
                producer = self.graph.producer_of(inp)
                assert producer is not None  # already validated above
                fname = _output_filename(producer, inp)
                local_path = self.staging_dir / fname
                wqt.specify_input_file(str(local_path), fname, cache=False)
            for out in saga_task.outputs:
                fname = _output_filename(name, out)
                local_path = self.staging_dir / fname
                wqt.specify_output_file(str(local_path), fname, cache=False)

            # *** The actual pinning. ***
            if self.require_features:
                wqt.specify_feature(feature_for(node))

            # Human-readable tag & category.
            wqt.specify_category(_category_name(node))
            wqt.specify_tag(name)

            task_id = manager.submit(wqt)
            wq_tasks[task_id] = name
            submitted_order.append(name)
            log(f"submitted {name!r} -> node={node} feature={feature_for(node)}")

        per_task_seconds: Dict[str, float] = {}
        completed_names: List[str] = []

        import time as _time

        t_start = _time.time()
        deadline = (t_start + timeout_seconds) if timeout_seconds else None
        while not manager.empty():
            wait = 5 if deadline is None else max(1, int(deadline - _time.time()))
            done = manager.wait(wait)
            if done is None:
                if deadline is not None and _time.time() >= deadline:
                    raise TimeoutError(
                        f"Work Queue run exceeded timeout of {timeout_seconds} seconds"
                    )
                continue
            name = wq_tasks.get(done.id, f"id={done.id}")
            if done.return_status != 0:
                log(
                    f"task {name!r} FAILED: "
                    f"return_status={done.return_status} "
                    f"result={getattr(done, 'result', 'unknown')}\n"
                    f"stdout:\n{getattr(done, 'output', '')}"
                )
                raise RuntimeError(f"task {name!r} failed on Work Queue")
            # ``total_submit_time`` and ``finish_time`` are in microseconds.
            submit_us = float(getattr(done, "total_submit_time", 0.0))
            finish_us = float(getattr(done, "finish_time", 0.0))
            per_task_seconds[name] = max(0.0, (finish_us - submit_us) / 1e6)
            completed_names.append(name)
            log(f"completed {name!r} in {per_task_seconds[name]:.2f}s")
        t_end = _time.time()

        outputs = {}
        for name, saga_task in self.graph.tasks.items():
            for out in saga_task.outputs:
                fname = _output_filename(name, out)
                outputs[f"{name}.{out}"] = self.staging_dir / fname

        return WorkQueueRunResult(
            makespan_seconds=t_end - t_start,
            per_task_seconds=per_task_seconds,
            output_files=outputs,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _task_to_node_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for node_name, scheduled in self.schedule.mapping.items():
            for st in scheduled:
                mapping[st.name] = node_name
        # Ensure every SagaGraph task is represented.
        for name in self.graph.tasks:
            if name not in mapping:
                raise ValueError(
                    f"task {name!r} was not assigned a node in the schedule"
                )
        return mapping
