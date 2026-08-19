"""Contention model: fan-in sharing, solo flows, same-node edges, and the
exclusive-vs-concurrent compute models."""

import pytest

from saga import Network, TaskGraph
from saga.contention import simulate_placement


@pytest.fixture
def net():
    return Network.create(
        nodes=[("a", 1.0), ("b", 1.0), ("c", 1.0)],
        edges=[("a", "b", 100.0), ("a", "c", 100.0), ("b", "c", 100.0),
               ("a", "a", 1e9), ("b", "b", 1e9), ("c", "c", 1e9)])


def test_fan_in_shares_the_receiving_nic(net):
    # Two producers on different nodes, 100 units each into one consumer.
    # Different links, so only an ingress resource makes them contend.
    tg = TaskGraph.create(
        tasks=[("p1", 1.0), ("p2", 1.0), ("sink", 1.0)],
        dependencies=[("p1", "sink", 100.0), ("p2", "sink", 100.0)])
    r = simulate_placement(net, tg, {"p1": "a", "p2": "b", "sink": "c"})
    assert len(r.transfers) == 2
    for f in r.transfers:
        assert f.duration == pytest.approx(2.0, abs=1e-6)   # half rate each
        assert f.mean_rate == pytest.approx(50.0, abs=1e-6)
    # The consumer waits for both.
    assert r.tasks["sink"].start == pytest.approx(3.0, abs=1e-6)


def test_solo_flow_gets_full_link_rate(net):
    tg = TaskGraph.create(tasks=[("p1", 1.0), ("sink", 1.0)],
                          dependencies=[("p1", "sink", 100.0)])
    r = simulate_placement(net, tg, {"p1": "a", "sink": "c"})
    assert r.transfers[0].duration == pytest.approx(1.0, abs=1e-6)


def test_same_node_edge_moves_nothing(net):
    tg = TaskGraph.create(tasks=[("p1", 1.0), ("sink", 1.0)],
                          dependencies=[("p1", "sink", 100.0)])
    r = simulate_placement(net, tg, {"p1": "c", "sink": "c"})
    assert r.transfers == []
    assert r.tasks["sink"].start == pytest.approx(r.tasks["p1"].end, abs=1e-6)


def test_concurrent_vs_exclusive_compute(net):
    tg = TaskGraph.create(tasks=[("x", 2.0), ("y", 2.0)], dependencies=[])
    conc = simulate_placement(net, tg, {"x": "a", "y": "a"},
                              task_cpu={"x": 1, "y": 1}, node_cpu={"a": 4})
    excl = simulate_placement(net, tg, {"x": "a", "y": "a"}, exclusive=True)
    assert conc.makespan == pytest.approx(2.0)   # both fit, run together
    assert excl.makespan == pytest.approx(4.0)   # serialised


def test_cpu_capacity_serialises_when_full(net):
    tg = TaskGraph.create(tasks=[("x", 2.0), ("y", 2.0)], dependencies=[])
    r = simulate_placement(net, tg, {"x": "a", "y": "a"},
                           task_cpu={"x": 3, "y": 3}, node_cpu={"a": 4})
    assert r.makespan == pytest.approx(4.0)      # 3+3 > 4, so they queue


def test_flow_speeds_up_when_a_competitor_finishes(net):
    # Unequal sizes into one consumer: the small flow finishes first and the
    # large one must accelerate for the remainder.
    tg = TaskGraph.create(
        tasks=[("p1", 1.0), ("p2", 1.0), ("sink", 1.0)],
        dependencies=[("p1", "sink", 100.0), ("p2", "sink", 50.0)])
    r = simulate_placement(net, tg, {"p1": "a", "p2": "b", "sink": "c"})
    big = next(f for f in r.transfers if f.src_task == "p1")
    small = next(f for f in r.transfers if f.src_task == "p2")
    assert small.duration == pytest.approx(1.0, abs=1e-6)   # 50 @ 50/s
    # 50 units at 50/s, then the remaining 50 alone at 100/s -> 1.5s total,
    # strictly faster than the 2.0s it would take if the split never lifted.
    assert big.duration == pytest.approx(1.5, abs=1e-6)


def test_synthetic_super_nodes_are_absorbed(net):
    # A multi-source graph makes SAGA inject __super_source__; the caller's
    # placement does not mention it and must not have to.
    tg = TaskGraph.create(
        tasks=[("s1", 1.0), ("s2", 1.0), ("sink", 1.0)],
        dependencies=[("s1", "sink", 10.0), ("s2", "sink", 10.0)])
    r = simulate_placement(net, tg, {"s1": "a", "s2": "b", "sink": "c"})
    assert set(r.tasks) == {"s1", "s2", "sink"}
