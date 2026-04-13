"""Task definitions for the Chameleon demo.

These are deliberately kept in a separate, importable module so that the
Work Queue wrapper on each worker can ``import tasks`` and find the
``@saga_task``-decorated callables.  (An inline ``__main__`` module won't
work because on the worker ``__main__`` is the wrapper itself, not the
driver script.)

Run via ``scripts/examples/dagprofiler_chameleon/main.py``.
"""

from __future__ import annotations

import numpy as np

from saga.execution import saga_task


@saga_task(
    inputs=[],
    outputs=["samples"],
    cost=lambda cfg: cfg["N"],  # O(N) generation
    output_sizes={"samples": lambda cfg: cfg["N"] * 8},  # float64 bytes
)
def generate(cfg):
    """Produce N random samples."""
    rng = np.random.default_rng(cfg.get("seed", 0))
    return {"samples": rng.standard_normal(cfg["N"])}


@saga_task(
    inputs=["samples"],
    outputs=["squared"],
    cost=lambda cfg: 10 * cfg["N"],  # 10 ops per sample
    output_sizes={"squared": lambda cfg: cfg["N"] * 8},
)
def square(samples, cfg):
    """Square every sample."""
    return {"squared": samples**2}


@saga_task(
    inputs=["samples"],
    outputs=["shifted"],
    cost=lambda cfg: 5 * cfg["N"],
    output_sizes={"shifted": lambda cfg: cfg["N"] * 8},
)
def shift(samples, cfg):
    """Shift every sample by +1 (a cheap, parallel branch)."""
    return {"shifted": samples + 1.0}


@saga_task(
    inputs=["squared", "shifted"],
    outputs=["stats"],
    cost=lambda cfg: 2 * cfg["N"],
    output_sizes={"stats": lambda cfg: 64},  # tiny summary
)
def combine(squared, shifted, cfg):
    """Fold the two parallel branches into summary stats."""
    both = squared + shifted
    return {"stats": {"mean": float(both.mean()), "std": float(both.std())}}
