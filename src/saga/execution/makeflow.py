"""Emit a Makeflow JSON workflow from a SAGA schedule.

Given a ``SagaGraph`` and a ``saga.Schedule``, this module writes a Makeflow
JSON file plus a small staging directory that Makeflow can execute against
(either locally or on a Work Queue cluster, via ``makeflow -T wq``).

The emitted workflow follows the Makeflow JSON schema documented at
https://cctools.readthedocs.io/en/latest/jx-workflow/jx/ — specifically the
subset with ``rules``, ``categories``, ``default_category``, ``environment``
and ``define``.

Each SAGA task becomes one Makeflow rule whose command invokes
``python -m saga.execution._wrapper`` (see ``_wrapper.py``).  Per-task data
dependencies become file inputs/outputs so Makeflow can order them correctly,
and the SAGA-chosen node for each task is recorded as a per-rule category.

Pinning tasks to specific workers at the Makeflow level is a soft hint —
Makeflow categories are ultimately about resource requirements, not worker
affinity.  For hard pinning on a Work Queue cluster, use
``saga.execution.workqueue.WorkQueueRunner`` which sets
``Task.specify_feature(...)`` per task.  The Makeflow JSON emitted here is
useful as a portable, inspectable record of the schedule and for local
``makeflow -T local`` runs.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from saga import Schedule

from .graph import SagaGraph


@dataclass
class MakeflowArtifacts:
    """File layout produced by ``emit_makeflow``."""

    root: pathlib.Path
    makeflow_json: pathlib.Path
    config_pickle: pathlib.Path

    def __str__(self) -> str:
        return str(self.makeflow_json)


def _output_filename(src_task: str, field: str) -> str:
    """Deterministic filename for one output field of a task."""
    return f"{src_task}__{field}.pkl"


def emit_makeflow(
    graph: SagaGraph,
    schedule: Schedule,
    output_dir: pathlib.Path | str,
    *,
    python_bin: str = "python3",
    task_module: Optional[str] = None,
    task_module_map: Optional[Dict[str, str]] = None,
    extra_inputs: Optional[List[str]] = None,
) -> MakeflowArtifacts:
    """Serialise a scheduled ``SagaGraph`` as Makeflow JSON.

    Args:
        graph: The SagaGraph whose tasks are being executed.
        schedule: A SAGA schedule (``task -> node``) produced by any
            ``Scheduler``.
        output_dir: Directory to create (or reuse) for the workflow and
            the staged config pickle.
        python_bin: Python interpreter to invoke on workers.
        task_module: Dotted path to the module that defines every task in
            ``graph`` (e.g. ``my_pkg.pipeline``).  Every task is imported
            with ``{task_module}.{task_name}``.  Either this or
            ``task_module_map`` must be set.
        task_module_map: Per-task override mapping ``task_name -> module``.
            Takes precedence over ``task_module`` when present.
        extra_inputs: Additional files to list as inputs of every rule
            (e.g. a user package tarball the workers need to unpack).

    Returns:
        ``MakeflowArtifacts`` describing the paths of the generated files.
    """
    if task_module is None and not task_module_map:
        raise ValueError(
            "specify either task_module= or task_module_map= so the worker "
            "wrapper knows which module to import"
        )

    root = pathlib.Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Stage the config as a pickle that every task depends on.
    config_path = root / "config.pkl"
    with config_path.open("wb") as fh:
        pickle.dump(dict(graph.config), fh, protocol=pickle.HIGHEST_PROTOCOL)

    # -- Categories: one per compute node used by the schedule. --
    categories: Dict[str, Dict[str, Any]] = {}
    for net_node in schedule.network.nodes:
        categories[_category_name(net_node.name)] = {
            # SAGA nodes carry only a ``speed``; we don't know physical
            # CPU counts, so emit a minimal resource block.  Real users can
            # override at run time.
            "resources": {"cores": 1}
        }

    # -- Rules: one per task in the graph. --
    # Map task name -> node name from the schedule.
    task_to_node: Dict[str, str] = {}
    for node_name, scheduled_tasks in schedule.mapping.items():
        for st in scheduled_tasks:
            task_to_node[st.name] = node_name

    # Skip SAGA's synthetic super-source/sink tasks — they carry no real work.
    synthetic = {"__super_source__", "__super_sink__"}

    rules: List[Dict[str, Any]] = []
    for t in graph.tasks.values():
        node = task_to_node.get(t.name)
        if node is None:
            raise ValueError(
                f"task {t.name!r} was not assigned a node in the schedule; "
                f"did you schedule the right TaskGraph?"
            )

        module = (task_module_map or {}).get(t.name, task_module)
        if module is None:
            raise ValueError(f"no module mapping available for task {t.name!r}")

        # Build the wrapper command.
        cmd_parts: List[str] = [
            python_bin,
            "-m",
            "saga.execution._wrapper",
            "--task-module",
            module,
            "--task-attr",
            t.name,
            "--config",
            config_path.name,
        ]

        # Each declared input of this task is fed by the matching output
        # of its producer.
        rule_inputs: List[str] = [config_path.name]
        if extra_inputs:
            rule_inputs.extend(extra_inputs)
        for inp in t.inputs:
            producer = graph.producer_of(inp)
            if producer is None or producer in synthetic:
                raise ValueError(
                    f"task {t.name!r} consumes {inp!r} with no real producer"
                )
            fname = _output_filename(producer, inp)
            cmd_parts += ["--input", f"{inp}={fname}"]
            rule_inputs.append(fname)

        # Outputs: one pickle file per declared output field.
        rule_outputs: List[str] = []
        for out in t.outputs:
            fname = _output_filename(t.name, out)
            cmd_parts += ["--output", f"{out}={fname}"]
            rule_outputs.append(fname)

        rule: Dict[str, Any] = {
            "command": " ".join(shlex.quote(p) for p in cmd_parts),
            "inputs": rule_inputs,
            "outputs": rule_outputs,
            "category": _category_name(node),
        }
        rules.append(rule)

    workflow: Dict[str, Any] = {
        "categories": categories,
        "rules": rules,
    }
    if categories:
        # Pick any category as a default so Makeflow doesn't complain if
        # future rules are added without one.
        workflow["default_category"] = next(iter(categories))

    makeflow_json = root / "workflow.json"
    makeflow_json.write_text(json.dumps(workflow, indent=2))

    return MakeflowArtifacts(
        root=root,
        makeflow_json=makeflow_json,
        config_pickle=config_path,
    )


def _category_name(node_name: str) -> str:
    """Normalise a node name for use as a Makeflow category token."""
    # Makeflow category names may contain most printable characters but we
    # play it safe and stick to a simple alphanumeric/underscore form.
    safe = "".join(c if c.isalnum() else "_" for c in node_name)
    return f"node_{safe}"
