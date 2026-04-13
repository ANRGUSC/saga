"""Adapter from ``dagprofiler`` profiles to SAGA task graphs.

``dagprofiler`` (https://github.com/ANRGUSC/dagprofiler) produces JSON profiles
that already carry exactly the two scalars SAGA needs: ``node_weights``
(per-task instruction estimates) and ``edge_weights`` (per-edge bits
transferred).  This module converts such a profile into a ``saga.TaskGraph``.

The dagprofiler JSON schema (summarised from the repo's ``profile.py``)::

    {
      "dag_structure": {
        "nodes": ["t1", "t2", ...],
        "edges": [{"source": "t1", "target": "t2"}, ...],
        ...
      },
      "node_weights":  {"t1": 100000, "t2": 250000, ...},
      "edge_weights":  {"t1->t2": 6400000, ...},   # bits
      "execution_metrics": [
        {"task": "t1", "compute_time_ms": 5.6, "bytes_in": ..., "bytes_out": ...},
        ...
      ],
      ...
    }

Two knobs allow different flavours of conversion:

* ``use_measured_time`` — if ``True``, use ``compute_time_ms`` from
  ``execution_metrics`` as the task cost (unit: milliseconds) instead of the
  analytic instruction estimate.  This is typically what you want after a
  real profiling run on a reference machine.
* ``edge_units`` — either ``"bits"`` (the raw value) or ``"bytes"`` (divide by
  8).  SAGA's network bandwidths don't care about the unit so long as task
  costs, edge sizes, and node/edge speeds use consistent units — but most
  users think in bytes.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Tuple, Union

from saga import TaskGraph


def _load_profile(source: Union[str, pathlib.Path, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(source, dict):
        return source
    path = pathlib.Path(source)
    return json.loads(path.read_text(encoding="utf-8"))


def profile_to_task_graph(
    source: Union[str, pathlib.Path, Dict[str, Any]],
    *,
    use_measured_time: bool = False,
    edge_units: str = "bytes",
) -> TaskGraph:
    """Convert a dagprofiler JSON profile into a SAGA ``TaskGraph``.

    Args:
        source: A JSON file path or a parsed dict.
        use_measured_time: If True, task cost comes from the measured
            ``compute_time_ms`` in ``execution_metrics`` (milliseconds) rather
            than the analytic ``node_weights`` (instructions).
        edge_units: ``"bits"`` (raw) or ``"bytes"`` (divide by 8).

    Returns:
        A ``TaskGraph`` with one node per dagprofiler task and one edge per
        dagprofiler edge.
    """
    if edge_units not in ("bits", "bytes"):
        raise ValueError(f"edge_units must be 'bits' or 'bytes', got {edge_units!r}")

    data = _load_profile(source)
    structure = data.get("dag_structure") or {}
    nodes: List[str] = list(structure.get("nodes", []))
    edge_pairs: List[Tuple[str, str]] = [
        (e["source"], e["target"]) for e in structure.get("edges", [])
    ]

    node_weights: Dict[str, float] = {
        k: float(v) for k, v in (data.get("node_weights") or {}).items()
    }
    edge_weights_raw: Dict[str, float] = {
        k: float(v) for k, v in (data.get("edge_weights") or {}).items()
    }

    measured: Dict[str, List[float]] = {}
    for m in data.get("execution_metrics") or []:
        # execution_metrics may contain multiple rows per task (loop runs);
        # take the mean so the cost is representative.
        name = m["task"]
        t = float(m.get("compute_time_ms", 0.0))
        measured.setdefault(name, []).append(t)
    measured_mean: Dict[str, float] = {
        name: (sum(vs) / len(vs)) if vs else 0.0 for name, vs in measured.items()
    }

    divisor = 8.0 if edge_units == "bytes" else 1.0

    tasks: List[Tuple[str, float]] = []
    for name in nodes:
        if use_measured_time:
            cost = measured_mean.get(name, 0.0)
        else:
            cost = node_weights.get(name, 0.0)
        tasks.append((name, max(float(cost), 1e-9)))

    deps: List[Tuple[str, str, float]] = []
    for src, dst in edge_pairs:
        key = f"{src}->{dst}"
        raw = edge_weights_raw.get(key, 0.0)
        size = raw / divisor
        deps.append((src, dst, max(float(size), 1e-9)))

    return TaskGraph.create(tasks=tasks, dependencies=deps)
