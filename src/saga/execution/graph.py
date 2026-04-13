"""A small Python-decorator notation for defining SAGA task graphs.

Users decorate plain Python functions with ``@saga_task`` and combine them into
a ``SagaGraph``.  The resulting graph can be converted to a ``saga.TaskGraph``
(for scheduling) or executed on a real compute platform via the emitters in
``saga.execution.makeflow`` / ``saga.execution.workqueue``.

The notation is intentionally minimal: it captures enough to feed SAGA's
scheduler (per-task cost, per-edge bytes) and enough to execute the graph on
a remote worker (the Python callable, serializable inputs/outputs).

Example::

    from saga.execution import saga_task, SagaGraph

    @saga_task(inputs=[], outputs=["x"], cost=lambda cfg: 1_000)
    def source(cfg):
        import numpy as np
        return {"x": np.arange(cfg["N"])}

    @saga_task(inputs=["x"], outputs=["y"],
               cost=lambda cfg: 100 * cfg["N"])
    def square(x, cfg):
        return {"y": x ** 2}

    @saga_task(inputs=["y"], outputs=["z"], cost=lambda cfg: cfg["N"])
    def summarize(y, cfg):
        return {"z": int(y.sum())}

    graph = SagaGraph(tasks=[source, square, summarize], config={"N": 10_000})
    task_graph = graph.to_task_graph()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from saga import TaskGraph


# A cost function takes a config dict and returns the task's compute cost
# (same units as SAGA's ``TaskGraphNode.cost``).  If omitted, a constant default
# is used.
CostFn = Callable[[Dict[str, Any]], float]

# A size function takes a config dict and returns the number of bytes that
# flow across an edge.  If omitted, a constant default is used.
SizeFn = Callable[[Dict[str, Any]], float]


@dataclass
class SagaTask:
    """A single task definition.

    Created by the ``@saga_task`` decorator.  A ``SagaTask`` carries the
    Python callable plus everything SAGA needs to reason about it: the input
    and output field names, a cost estimator, and optional per-output data
    sizes.
    """

    name: str
    fn: Callable[..., Dict[str, Any]]
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    cost: CostFn = field(default=lambda cfg: 1.0)
    # Map of output field name -> size function.  The emitted edge for
    # ``(self -> downstream)`` is the sum of the sizes of the matched fields.
    output_sizes: Dict[str, SizeFn] = field(default_factory=dict)

    def estimate_cost(self, config: Dict[str, Any]) -> float:
        value = float(self.cost(config))
        # SAGA requires strictly positive task costs; guard against 0/negative.
        return max(value, 1e-9)


def saga_task(
    *,
    inputs: Iterable[str],
    outputs: Iterable[str],
    cost: Optional[CostFn] = None,
    output_sizes: Optional[Dict[str, SizeFn]] = None,
    name: Optional[str] = None,
) -> Callable[[Callable[..., Dict[str, Any]]], SagaTask]:
    """Decorate a Python function as a SAGA task.

    Args:
        inputs: Names of input fields the task consumes.  Each name must be
            produced by an upstream task's ``outputs``.
        outputs: Names of output fields the task produces.
        cost: Optional cost model ``cfg -> float``.  The returned value is
            used as ``TaskGraphNode.cost`` and has the same units as the
            node speeds in the target ``Network``.
        output_sizes: Optional map ``output_name -> (cfg -> bytes)`` giving
            the data size on each outgoing edge.  Unspecified fields default
            to 1 byte.
        name: Optional task name (defaults to the function's ``__name__``).

    Returns:
        A decorator that turns the function into a ``SagaTask`` instance.
    """

    _inputs = tuple(inputs)
    _outputs = tuple(outputs)
    _cost: CostFn = cost or (lambda cfg: 1.0)
    _sizes = dict(output_sizes or {})

    def decorator(fn: Callable[..., Dict[str, Any]]) -> SagaTask:
        return SagaTask(
            name=name or fn.__name__,
            fn=fn,
            inputs=_inputs,
            outputs=_outputs,
            cost=_cost,
            output_sizes=_sizes,
        )

    return decorator


class SagaGraph:
    """A collection of ``SagaTask`` instances with a resolved DAG structure.

    Edges are inferred by matching each task's ``inputs`` to the unique
    upstream producer of that output name.  Edges may also be supplied
    explicitly, in which case the ``data`` for that edge is the intersection
    of producer-outputs and consumer-inputs.
    """

    def __init__(
        self,
        tasks: Iterable[SagaTask],
        config: Optional[Dict[str, Any]] = None,
        edges: Optional[Iterable[Tuple[str, str]]] = None,
    ):
        self.tasks: Dict[str, SagaTask] = {}
        for t in tasks:
            if t.name in self.tasks:
                raise ValueError(f"Duplicate task name: {t.name}")
            self.tasks[t.name] = t

        self.config: Dict[str, Any] = dict(config or {})

        # output-name -> producing task
        self._producers: Dict[str, str] = {}
        for t in self.tasks.values():
            for out in t.outputs:
                if out in self._producers:
                    raise ValueError(
                        f"Output {out!r} is produced by both "
                        f"{self._producers[out]!r} and {t.name!r}; "
                        f"rename one of them."
                    )
                self._producers[out] = t.name

        # edges: list of (source_task, target_task, shared_field_names)
        self._edges: List[Tuple[str, str, Tuple[str, ...]]] = []
        explicit_edges = list(edges) if edges is not None else None
        self._build_edges(explicit_edges)

    def _build_edges(self, explicit: Optional[List[Tuple[str, str]]]) -> None:
        """Infer (or validate) edges between tasks."""
        seen: Dict[Tuple[str, str], List[str]] = {}

        if explicit is None:
            # Infer from each consumer's inputs.
            for t in self.tasks.values():
                for inp in t.inputs:
                    if inp not in self._producers:
                        raise ValueError(
                            f"Task {t.name!r} consumes input {inp!r}, "
                            f"but no task produces it."
                        )
                    src = self._producers[inp]
                    if src == t.name:
                        raise ValueError(
                            f"Task {t.name!r} consumes its own output {inp!r}"
                        )
                    seen.setdefault((src, t.name), []).append(inp)
        else:
            for src, dst in explicit:
                if src not in self.tasks or dst not in self.tasks:
                    raise ValueError(
                        f"Edge references unknown task(s): {src!r} -> {dst!r}"
                    )
                shared = [
                    f for f in self.tasks[src].outputs if f in self.tasks[dst].inputs
                ]
                seen[(src, dst)] = shared

        self._edges = [(src, dst, tuple(fields)) for (src, dst), fields in seen.items()]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def edges(self) -> List[Tuple[str, str, Tuple[str, ...]]]:
        """The resolved DAG edges as ``(src, dst, shared_field_names)``."""
        return list(self._edges)

    def get_task(self, name: str) -> SagaTask:
        return self.tasks[name]

    def producer_of(self, output_name: str) -> Optional[str]:
        return self._producers.get(output_name)

    # ------------------------------------------------------------------
    # Conversion to SAGA TaskGraph
    # ------------------------------------------------------------------

    def to_task_graph(
        self,
        node_cost_overrides: Optional[Dict[str, float]] = None,
        edge_size_overrides: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> TaskGraph:
        """Build a ``saga.TaskGraph`` using declared cost/size models.

        Args:
            node_cost_overrides: Map ``task_name -> cost`` to override any
                declared cost models (useful after profiling).
            edge_size_overrides: Map ``(src, dst) -> bytes`` to override
                edge sizes (useful after profiling).

        Returns:
            A SAGA ``TaskGraph``.  If the graph has multiple sources or sinks,
            ``TaskGraph.create`` will wrap them with ``__super_source__`` /
            ``__super_sink__`` automatically.
        """
        node_cost_overrides = dict(node_cost_overrides or {})
        edge_size_overrides = dict(edge_size_overrides or {})

        task_items: List[Tuple[str, float]] = []
        for name, t in self.tasks.items():
            cost = node_cost_overrides.get(name, t.estimate_cost(self.config))
            task_items.append((name, float(cost)))

        dep_items: List[Tuple[str, str, float]] = []
        for src, dst, fields in self._edges:
            if (src, dst) in edge_size_overrides:
                size = float(edge_size_overrides[(src, dst)])
            else:
                src_task = self.tasks[src]
                # Sum the bytes of each shared field, defaulting to 1 byte
                # when the user didn't declare a size model.
                total = 0.0
                for field_name in fields:
                    fn = src_task.output_sizes.get(field_name)
                    total += float(fn(self.config)) if fn is not None else 1.0
                size = total
            dep_items.append((src, dst, max(size, 1e-9)))

        return TaskGraph.create(tasks=task_items, dependencies=dep_items)
