"""Regression tests for Network.scale_to_ccr.

Guards against a bug where scale_to_ccr overwrote every edge with a single
average speed, destroying link heterogeneity (all edges ended up equal) and
clobbering self-loops. See issue #50.
"""

import math

import pytest

from saga import Network, TaskGraph

EPS = 1e-9


def _make_instance():
    """A heterogeneous network and a simple chain task graph."""
    network = Network.create(
        nodes=[("A", 1.0), ("B", 2.0), ("C", 4.0)],
        edges=[("A", "B", 2.0), ("A", "C", 3.0), ("B", "C", 5.0)],
    )
    task_graph = TaskGraph.create(
        tasks=[("t1", 1.0), ("t2", 2.0), ("t3", 3.0)],
        dependencies=[("t1", "t2", 10.0), ("t2", "t3", 20.0)],
    )
    return network, task_graph


def _inter_node_speeds(network):
    return sorted(e.speed for e in network.edges if e.source != e.target)


def _measured_ccr(network, task_graph):
    """CCR measured the way the codebase defines it (mean inter-node link speed)."""
    avg_node_speed = sum(n.speed for n in network.nodes) / len(network.nodes)
    avg_task_cost = sum(t.cost for t in task_graph.tasks) / len(task_graph.tasks)
    avg_data_size = sum(d.size for d in task_graph.dependencies) / len(
        task_graph.dependencies
    )
    inter = [e.speed for e in network.edges if e.source != e.target]
    mean_link_speed = sum(inter) / len(inter)
    avg_comp_time = avg_task_cost / avg_node_speed
    return (avg_data_size / mean_link_speed) / avg_comp_time


@pytest.mark.parametrize("target_ccr", [0.25, 1.0, 4.0])
def test_scale_to_ccr_hits_target(target_ccr):
    network, task_graph = _make_instance()
    scaled = network.scale_to_ccr(task_graph, target_ccr)
    assert math.isclose(_measured_ccr(scaled, task_graph), target_ccr, rel_tol=1e-6)


def test_scale_to_ccr_preserves_heterogeneity_and_ratios():
    network, task_graph = _make_instance()
    before = _inter_node_speeds(network)
    scaled = network.scale_to_ccr(task_graph, target_ccr=1.0)
    after = _inter_node_speeds(scaled)

    # Links stay distinct rather than collapsing to one value.
    assert len(set(after)) == len(set(before)) > 1
    # Relative heterogeneity is preserved: every link scaled by the same factor.
    ratios = [a / b for a, b in zip(after, before)]
    assert all(math.isclose(r, ratios[0], rel_tol=1e-9) for r in ratios)


def test_scale_to_ccr_leaves_self_loops_untouched():
    network, task_graph = _make_instance()
    scaled = network.scale_to_ccr(task_graph, target_ccr=1.0)
    for edge in scaled.edges:
        if edge.source == edge.target:
            assert edge.speed == math.inf


def test_scale_to_ccr_rejects_nonpositive_target():
    network, task_graph = _make_instance()
    with pytest.raises(ValueError):
        network.scale_to_ccr(task_graph, target_ccr=0.0)


def test_scale_to_ccr_rejects_network_without_inter_node_links():
    # A single-node network has only a self-loop, so the target CCR is
    # unachievable and scaling should fail fast rather than no-op.
    _, task_graph = _make_instance()
    network = Network.create(nodes=[("A", 1.0)], edges=[])
    with pytest.raises(ValueError):
        network.scale_to_ccr(task_graph, target_ccr=1.0)
