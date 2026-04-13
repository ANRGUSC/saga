"""Stand-alone task-runner invoked on workers.

A SAGA task, defined in user Python code with ``@saga_task``, is executed on a
remote worker as follows:

1. The driver serialises each input field to a pickle file.
2. The driver launches a subprocess on the worker:

       python -m saga.execution._wrapper \\
            --task-module my_pkg.tasks \\
            --task-attr   my_task \\
            --config      config.pkl \\
            --input       x=x.pkl \\
            --input       y=y.pkl \\
            --output      z=z.pkl

3. This module imports ``my_pkg.tasks.my_task`` (a ``SagaTask``), loads the
   pickled inputs, invokes the callable, and pickles the named outputs to the
   requested paths.

This design keeps the worker side completely generic: every SAGA task, on every
platform (Makeflow, Work Queue, plain bash-over-ssh) runs exactly the same
wrapper command.
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import sys
from typing import Any, Dict, List, Tuple


def _parse_kv(items: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected KEY=PATH but got {item!r}")
        key, _, path = item.partition("=")
        pairs.append((key, path))
    return pairs


def _import_task(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise SystemExit(
            f"module {module_name!r} has no attribute {attr_name!r}"
        ) from exc


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Run a single SAGA task.")
    ap.add_argument(
        "--task-module", required=True, help="dotted import path of the task's module"
    )
    ap.add_argument(
        "--task-attr",
        required=True,
        help="attribute name of the SagaTask within the module",
    )
    ap.add_argument(
        "--config",
        required=False,
        default=None,
        help="pickle file containing the config dict (optional)",
    )
    ap.add_argument(
        "--input",
        action="append",
        default=[],
        help="NAME=PATH for each serialised input",
    )
    ap.add_argument(
        "--output",
        action="append",
        default=[],
        help="NAME=PATH for each serialised output",
    )
    args = ap.parse_args(argv)

    saga_task = _import_task(args.task_module, args.task_attr)
    if not hasattr(saga_task, "fn") or not hasattr(saga_task, "inputs"):
        raise SystemExit(
            f"{args.task_module}.{args.task_attr} is not a SagaTask "
            f"(did you forget the @saga_task decorator?)"
        )

    config: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "rb") as fh:
            config = pickle.load(fh)

    inputs: Dict[str, Any] = {}
    for name, path in _parse_kv(args.input):
        with open(path, "rb") as fh:
            inputs[name] = pickle.load(fh)

    # Check all declared inputs are satisfied.
    missing = set(saga_task.inputs) - set(inputs)
    if missing:
        raise SystemExit(
            f"task {saga_task.name!r} is missing inputs: {sorted(missing)}"
        )

    # Invoke the user's function.
    result = saga_task.fn(cfg=config, **inputs)
    if not isinstance(result, dict):
        raise SystemExit(
            f"task {saga_task.name!r} returned {type(result).__name__}, "
            f"expected dict of output fields"
        )

    # Write every requested output.
    declared = set(saga_task.outputs)
    for name, path in _parse_kv(args.output):
        if name not in declared:
            raise SystemExit(
                f"task {saga_task.name!r} did not declare output {name!r}; "
                f"declared: {sorted(declared)}"
            )
        if name not in result:
            raise SystemExit(f"task {saga_task.name!r} did not produce output {name!r}")
        with open(path, "wb") as fh:
            pickle.dump(result[name], fh, protocol=pickle.HIGHEST_PROTOCOL)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
