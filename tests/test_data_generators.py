"""Regression tests for the random task-graph/network generators.

These guard against a clamp bug in the default weight functions where an
inverted ``min``/``max`` made every generated weight collapse to ``2.0``,
producing fully degenerate (tie-everywhere) instances. See issue #58.
"""

import random

import pytest

from saga.schedulers.data.random import (
    _default_dependency_weight,
    _default_task_weight,
    gen_in_trees,
    gen_out_trees,
    gen_parallel_chains,
    gen_random_networks,
)

EPS = 1e-9


@pytest.mark.parametrize(
    "weight_fn",
    [
        lambda: _default_task_weight("T0"),
        lambda: _default_dependency_weight("T0", "T1"),
    ],
    ids=["task_weight", "dependency_weight"],
)
def test_default_weight_is_clamped_and_not_constant(weight_fn):
    """Default weights stay within [EPS, 2] and vary across draws."""
    random.seed(0)
    samples = [weight_fn() for _ in range(1000)]

    assert all(EPS <= w <= 2 for w in samples), "weights escaped the [EPS, 2] clamp"
    # The bug pinned every draw to exactly 2.0; a healthy generator spreads out.
    assert len(set(samples)) > 1, "weights are constant (degenerate instance)"
    assert any(abs(w - 2.0) > EPS for w in samples), "every weight collapsed to 2.0"


def test_generated_task_graphs_are_not_degenerate():
    """In/out trees and parallel chains have varied, non-degenerate costs."""
    random.seed(0)
    graphs = (
        gen_out_trees(1, num_levels=3, branching_factor=2)
        + gen_in_trees(1, num_levels=3, branching_factor=2)
        + gen_parallel_chains(1, num_chains=3, chain_length=3)
    )
    for task_graph in graphs:
        # Ignore the near-zero source/sink placeholder tasks (weight EPS).
        costs = [task.cost for task in task_graph.tasks if task.cost > EPS]
        assert len(set(costs)) > 1, "task costs are constant (degenerate instance)"


def test_generated_networks_are_not_degenerate():
    """Random networks have varied node speeds rather than a single value."""
    random.seed(0)
    network = gen_random_networks(1, num_nodes=5)[0]
    speeds = [node.speed for node in network.nodes]
    assert len(set(speeds)) > 1, "node speeds are constant (degenerate instance)"
