"""Tests for per-task placement constraints carried on the Schedule.

Constraints are per-instance data on the Schedule (`node_constraints`), passed either
to Schedule directly or through ParametricScheduler.schedule(node_constraints=...). A
task listed in the map may only be placed on a node in its allowed set; tasks absent from
the map are unconstrained. The greedy insert respects them (overriding both the caller's
`nodes` argument and the critical-path pin), and Schedule.add_task rejects any violating
placement with ConstraintViolation.
"""

import pytest

from saga import ConstraintViolation, Network, Schedule, ScheduledTask, TaskGraph
from saga.schedulers import FastestNodeScheduler
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    GreedyInsert,
    GreedyInsertCompareFuncs,
    UpwardRanking,
)


@pytest.fixture
def three_node_network():
    # v3 is the fastest, so an unconstrained EFT placement prefers it.
    return Network.create(
        nodes=[("v1", 1.0), ("v2", 1.0), ("v3", 10.0)],
        edges=[
            ("v1", "v2", 1.0),
            ("v1", "v3", 1.0),
            ("v2", "v3", 1.0),
        ],
    )


@pytest.fixture
def diamond_task_graph():
    """a -> {b, c} -> d"""
    return TaskGraph.create(
        tasks=[("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)],
        dependencies=[
            ("a", "b", 1.0),
            ("a", "c", 1.0),
            ("b", "d", 1.0),
            ("c", "d", 1.0),
        ],
    )


def _node_of(schedule: Schedule, task_name: str) -> str:
    for node, tasks in schedule.mapping.items():
        for t in tasks:
            if t.name == task_name:
                return node
    raise AssertionError(f"{task_name} not scheduled")


def test_unconstrained_prefers_fastest(three_node_network, diamond_task_graph):
    sched = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(compare=GreedyInsertCompareFuncs.EFT),
    )
    schedule = sched.schedule(three_node_network, diamond_task_graph)
    # With no constraints, EFT sends the root to the fast node.
    assert _node_of(schedule, "a") == "v3"


def test_constraint_pins_task_to_slow_node(three_node_network, diamond_task_graph):
    sched = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(compare=GreedyInsertCompareFuncs.EFT),
    )
    schedule = sched.schedule(
        three_node_network,
        diamond_task_graph,
        node_constraints={"a": {"v1"}, "d": {"v2"}},
    )
    assert _node_of(schedule, "a") == "v1"
    assert _node_of(schedule, "d") == "v2"
    # Unconstrained tasks are still free to take the fast node. b and c are symmetric,
    # so which one lands there is tie-break dependent, but one of them must.
    assert "v3" in {_node_of(schedule, "b"), _node_of(schedule, "c")}


def test_constraint_overrides_critical_path_pin(three_node_network, diamond_task_graph):
    # critical_path would pin the critical task to the fastest node (v3); the constraint
    # must win and keep it off v3.
    sched = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(
            compare=GreedyInsertCompareFuncs.EFT, critical_path=True
        ),
    )
    schedule = sched.schedule(
        three_node_network, diamond_task_graph, node_constraints={"a": {"v1", "v2"}}
    )
    assert _node_of(schedule, "a") in {"v1", "v2"}


def test_infeasible_constraint_raises(three_node_network, diamond_task_graph):
    sched = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(compare=GreedyInsertCompareFuncs.EFT),
    )
    with pytest.raises(ConstraintViolation):
        sched.schedule(
            three_node_network,
            diamond_task_graph,
            node_constraints={"a": {"does-not-exist"}},
        )


def test_constraint_composes_with_nodes_argument(three_node_network, diamond_task_graph):
    # The caller restricts to {v2, v3}; the task constraint is {v1, v3}; intersection is v3.
    schedule = Schedule(
        diamond_task_graph, three_node_network, node_constraints={"a": {"v1", "v3"}}
    )
    insert = GreedyInsert(compare=GreedyInsertCompareFuncs.EFT)
    placed = insert.call(
        three_node_network,
        diamond_task_graph,
        schedule,
        "a",
        nodes=["v2", "v3"],
    )
    assert placed.node == "v3"


def test_earliest_start_time_rejects_forbidden_node(three_node_network, diamond_task_graph):
    schedule = Schedule(
        diamond_task_graph, three_node_network, node_constraints={"a": {"v1"}}
    )
    # Querying a start time on a forbidden node is an invalid query.
    with pytest.raises(ConstraintViolation):
        schedule.get_earliest_start_time("a", "v2")
    # The allowed node is fine.
    assert schedule.get_earliest_start_time("a", "v1") == 0.0
    # Once a is placed, an unconstrained successor may be queried on any node.
    schedule.add_task(ScheduledTask(node="v1", name="a", start=0.0, end=1.0))
    assert schedule.get_earliest_start_time("b", "v3") >= 1.0


def test_fastest_node_unconstrained_uses_single_fastest(
    three_node_network, diamond_task_graph
):
    schedule = FastestNodeScheduler().schedule(three_node_network, diamond_task_graph)
    # Classic baseline: every task on the single fastest node.
    assert {_node_of(schedule, t) for t in ("a", "b", "c", "d")} == {"v3"}


def test_fastest_node_respects_constraints(three_node_network, diamond_task_graph):
    schedule = FastestNodeScheduler().schedule(
        three_node_network,
        diamond_task_graph,
        node_constraints={"a": {"v1"}, "d": {"v1", "v2"}},
    )
    # Pinned tasks take the fastest node in their own allowed set.
    assert _node_of(schedule, "a") == "v1"
    assert _node_of(schedule, "d") in {"v1", "v2"}
    # Unconstrained tasks still take the global fastest node.
    assert _node_of(schedule, "b") == "v3"
    assert _node_of(schedule, "c") == "v3"


def test_add_task_rejects_violating_placement(three_node_network, diamond_task_graph):
    schedule = Schedule(
        diamond_task_graph, three_node_network, node_constraints={"a": {"v1"}}
    )
    # Placing "a" on v2 violates its constraint.
    with pytest.raises(ConstraintViolation):
        schedule.add_task(ScheduledTask(node="v2", name="a", start=0.0, end=1.0))
    # Placing it on v1 is fine, and an unconstrained task goes anywhere.
    schedule.add_task(ScheduledTask(node="v1", name="a", start=0.0, end=1.0))
    schedule.add_task(ScheduledTask(node="v3", name="b", start=2.0, end=2.1))
