"""Tests for per-task placement constraints on StochasticSchedule.

Mirrors the deterministic constraint tests: constraints are per-instance data carried on
the StochasticSchedule (`node_constraints`). add_task rejects a violating placement with
ConstraintViolation, determinize propagates the constraints to the sampled deterministic
Schedule, and planning through EstimateStochasticScheduler honors them.
"""

import pytest

from saga import ConstraintViolation
from saga.stochastic import (
    StochasticNetwork,
    StochasticTaskGraph,
    StochasticSchedule,
    StochasticScheduledTask,
)
from saga.schedulers.stochastic.estimate_stochastic_scheduler import (
    EstimateStochasticScheduler,
)
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    GreedyInsert,
    GreedyInsertCompareFuncs,
    UpwardRanking,
)


@pytest.fixture
def three_node_network():
    return StochasticNetwork.create(
        nodes=[("v1", 1.0), ("v2", 1.0), ("v3", 10.0)],
        edges=[("v1", "v2", 1.0), ("v1", "v3", 1.0), ("v2", "v3", 1.0)],
    )


@pytest.fixture
def diamond_task_graph():
    """a -> {b, c} -> d"""
    return StochasticTaskGraph.create(
        tasks=[("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)],
        dependencies=[
            ("a", "b", 1.0),
            ("a", "c", 1.0),
            ("b", "d", 1.0),
            ("c", "d", 1.0),
        ],
    )


def _node_of(schedule, task_name):
    for node, tasks in schedule.mapping.items():
        for t in tasks:
            if t.name == task_name:
                return node
    raise AssertionError(f"{task_name} not scheduled")


def test_add_task_rejects_violating_placement(three_node_network, diamond_task_graph):
    schedule = StochasticSchedule(
        diamond_task_graph, three_node_network, node_constraints={"a": {"v1"}}
    )
    with pytest.raises(ConstraintViolation):
        schedule.add_task(StochasticScheduledTask(node="v2", name="a", rank=0.0))
    # Allowed placement and unconstrained tasks are fine.
    schedule.add_task(StochasticScheduledTask(node="v1", name="a", rank=0.0))
    schedule.add_task(StochasticScheduledTask(node="v3", name="b", rank=1.0))


def test_determinize_propagates_constraints(three_node_network, diamond_task_graph):
    inner = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(compare=GreedyInsertCompareFuncs.EFT),
    )
    scheduler = EstimateStochasticScheduler(inner, estimate=lambda rv: rv.mean())
    stoch_schedule, _, _ = scheduler.schedule(
        three_node_network, diamond_task_graph, node_constraints={"a": {"v1"}}
    )
    det = stoch_schedule.sample()
    # The sampled deterministic schedule carries the same constraints, and honors them.
    assert det.allowed_nodes("a") == {"v1"}
    assert _node_of(det, "a") == "v1"


def test_estimate_scheduler_honors_constraints(three_node_network, diamond_task_graph):
    inner = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(compare=GreedyInsertCompareFuncs.EFT),
    )
    scheduler = EstimateStochasticScheduler(inner, estimate=lambda rv: rv.mean())
    stoch_schedule, _, _ = scheduler.schedule(
        three_node_network,
        diamond_task_graph,
        node_constraints={"a": {"v1"}, "d": {"v2"}},
    )
    assert _node_of(stoch_schedule, "a") == "v1"
    assert _node_of(stoch_schedule, "d") == "v2"
    # The resulting stochastic schedule enforces the constraints on further edits.
    assert stoch_schedule.allowed_nodes("a") == {"v1"}
