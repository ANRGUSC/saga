"""Tests for overlap policies and how a Schedule consults them.

A Schedule normally forbids two tasks from sharing a node/time slot. That rule is
delegated to an OverlapPolicy: NoOverlapPolicy (the default) forbids all overlap,
while AllowOverlapPolicy permits an explicit set of task pairs. A ConditionalTaskGraph
supplies an AllowOverlapPolicy built from its mutual-exclusion graph, so tasks on
mutually exclusive branches may overlap; every other task graph uses NoOverlapPolicy.
"""

import pytest

from saga import Network, Schedule, ScheduledTask, TaskGraph
from saga.overlap import AllowOverlapPolicy, NoOverlapPolicy, OverlapPolicy
from saga.conditional import ConditionalTaskGraph, ConditionalTaskGraphEdge


# --------------------------------------------------------------------------- #
# Policy unit tests
# --------------------------------------------------------------------------- #


def test_no_overlap_policy_denies_everything():
    policy = NoOverlapPolicy()
    assert policy.can_overlap("A", "B") is False
    assert policy.can_overlap("A", "A") is False


def test_allow_overlap_policy_is_symmetric():
    policy = AllowOverlapPolicy.from_pairs([("B", "C")])
    assert policy.can_overlap("B", "C") is True
    assert policy.can_overlap("C", "B") is True


def test_allow_overlap_policy_denies_unlisted_and_self_pairs():
    policy = AllowOverlapPolicy.from_pairs([("B", "C"), ("D", "D")])
    assert policy.can_overlap("A", "B") is False  # unlisted task
    assert policy.can_overlap("B", "D") is False  # unlisted pair
    assert policy.can_overlap("D", "D") is False  # self-pairs are ignored


def test_allow_overlap_policy_is_not_transitive():
    # The relation is pairwise: B~C and C~E allowed must NOT imply B~E.
    policy = AllowOverlapPolicy.from_pairs([("B", "C"), ("C", "E")])
    assert policy.can_overlap("B", "C") is True
    assert policy.can_overlap("C", "E") is True
    assert policy.can_overlap("B", "E") is False


def test_empty_allow_overlap_policy_denies_everything():
    policy = AllowOverlapPolicy.from_pairs([])
    assert policy.can_overlap("A", "B") is False


def test_policies_satisfy_the_protocol():
    assert isinstance(NoOverlapPolicy(), OverlapPolicy)
    assert isinstance(AllowOverlapPolicy.from_pairs([("A", "B")]), OverlapPolicy)


# --------------------------------------------------------------------------- #
# Fixtures for the wiring tests
# --------------------------------------------------------------------------- #


@pytest.fixture
def one_node_network():
    return Network.create(nodes=[("n0", 1.0)], edges=[])


@pytest.fixture
def plain_diamond():
    """a -> {b, c} -> d, no conditional edges."""
    return TaskGraph.create(
        tasks=[("A", 1.0), ("B", 1.0), ("C", 1.0), ("D", 1.0)],
        dependencies=[
            ("A", "B", 1.0),
            ("A", "C", 1.0),
            ("B", "D", 1.0),
            ("C", "D", 1.0),
        ],
    )


@pytest.fixture
def conditional_diamond():
    """A branches conditionally to B or C; both lead to D.

    Traces are {A, B, D} and {A, C, D}, so B and C are mutually exclusive.
    """
    return ConditionalTaskGraph.create(
        tasks=[("A", 1.0), ("B", 1.0), ("C", 1.0), ("D", 1.0)],
        dependencies=[
            ConditionalTaskGraphEdge(source="A", target="B", size=1.0, probability=0.5),
            ConditionalTaskGraphEdge(source="A", target="C", size=1.0, probability=0.5),
            ConditionalTaskGraphEdge(source="B", target="D", size=1.0),
            ConditionalTaskGraphEdge(source="C", target="D", size=1.0),
        ],
    )


def _task(name: str, start: float, end: float, node: str = "n0") -> ScheduledTask:
    return ScheduledTask(node=node, name=name, start=start, end=end)


# --------------------------------------------------------------------------- #
# Base TaskGraph: classic (no overlap) behavior
# --------------------------------------------------------------------------- #


def test_plain_task_graph_uses_no_overlap_policy(plain_diamond):
    assert isinstance(plain_diamond.overlap_policy(), NoOverlapPolicy)


def test_plain_schedule_rejects_time_overlap(one_node_network, plain_diamond):
    schedule = Schedule(plain_diamond, one_node_network)
    schedule.add_task(_task("A", 0.0, 1.0))
    with pytest.raises(ValueError):
        schedule.add_task(_task("B", 0.5, 1.5))  # overlaps A on n0


# --------------------------------------------------------------------------- #
# ConditionalTaskGraph: overlap for mutually exclusive tasks
# --------------------------------------------------------------------------- #


def test_conditional_overlap_policy_matches_mutual_exclusion_graph(conditional_diamond):
    policy = conditional_diamond.overlap_policy()
    assert isinstance(policy, AllowOverlapPolicy)

    expected_pairs = {frozenset(edge) for edge in conditional_diamond.build_mutual_exclusion_graph().edges}
    assert expected_pairs == {frozenset(("B", "C"))}

    # B and C are the only mutually exclusive pair; everything else must be denied.
    assert policy.can_overlap("B", "C") is True
    assert policy.can_overlap("C", "B") is True
    assert policy.can_overlap("A", "B") is False
    assert policy.can_overlap("B", "D") is False


def test_conditional_schedule_allows_mutually_exclusive_overlap(
    one_node_network, conditional_diamond
):
    schedule = Schedule(conditional_diamond, one_node_network)
    schedule.add_task(_task("B", 0.0, 1.0))
    schedule.add_task(_task("C", 0.0, 1.0))  # mutually exclusive with B: allowed
    names_on_node = {t.name for t in schedule.mapping["n0"]}
    assert names_on_node == {"B", "C"}


def test_conditional_schedule_still_rejects_non_exclusive_overlap(
    one_node_network, conditional_diamond
):
    schedule = Schedule(conditional_diamond, one_node_network)
    schedule.add_task(_task("A", 0.0, 1.0))
    with pytest.raises(ValueError):
        schedule.add_task(_task("D", 0.5, 1.5))  # A and D co-occur: not allowed


def test_overlap_policy_override_forces_classic_behavior(
    one_node_network, conditional_diamond
):
    # Passing an explicit policy overrides what the task graph would supply.
    schedule = Schedule(
        conditional_diamond, one_node_network, overlap_policy=NoOverlapPolicy()
    )
    schedule.add_task(_task("B", 0.0, 1.0))
    with pytest.raises(ValueError):
        schedule.add_task(_task("C", 0.0, 1.0))  # would overlap; override forbids it
