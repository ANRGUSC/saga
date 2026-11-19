"""Tests for the randomized parametric components (RandomTopological, RandomInsert)."""

import random

import networkx as nx
import pytest

from saga import TaskGraph
from saga.schedulers.data.random import gen_random_networks
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    RandomInsert,
    RandomTopological,
    insert_funcs,
)


def _task_graph_from_edges(edges) -> TaskGraph:
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    for node in graph.nodes:
        graph.nodes[node]["weight"] = 1.0
    for edge in graph.edges:
        graph.edges[edge]["weight"] = 1.0
    return TaskGraph.from_nx(graph)


def _is_topological(order, task_graph: TaskGraph) -> bool:
    position = {name: i for i, name in enumerate(order)}
    return len(order) == len(task_graph.tasks) and all(
        position[dep.source] < position[dep.target] for dep in task_graph.dependencies
    )


@pytest.fixture
def unequal_paths_graph() -> TaskGraph:
    """Two paths of different length converge on D. A level-synchronous shuffle that
    checks readiness against a set it mutates can place D before its parent F here."""
    return _task_graph_from_edges(
        [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("A", "E"), ("E", "F"), ("F", "D")]
    )


@pytest.fixture
def large_random_dag() -> TaskGraph:
    graph = nx.gnp_random_graph(400, 0.02, directed=True, seed=1)
    edges = [(str(u), str(v)) for (u, v) in graph.edges() if u < v]
    return _task_graph_from_edges(edges)


def test_random_topological_is_valid_on_unequal_paths(unequal_paths_graph):
    network = gen_random_networks(num=1, num_nodes=3)[0]
    for seed in range(200):
        order = RandomTopological(seed=seed).call(network, unequal_paths_graph)
        assert _is_topological(order, unequal_paths_graph), order


def test_random_topological_reaches_every_order(unequal_paths_graph):
    network = gen_random_networks(num=1, num_nodes=3)[0]
    valid = {
        tuple(t.name for t in order) for order in unequal_paths_graph.all_topological_sorts()
    }
    seen = {
        tuple(RandomTopological(seed=seed).call(network, unequal_paths_graph))
        for seed in range(2000)
    }
    assert seen == valid


def test_random_topological_is_valid_on_large_dag(large_random_dag):
    network = gen_random_networks(num=1, num_nodes=4)[0]
    for seed in range(5):
        order = RandomTopological(seed=seed).call(network, large_random_dag)
        assert _is_topological(order, large_random_dag)


def test_random_topological_seed_is_deterministic(large_random_dag):
    network = gen_random_networks(num=1, num_nodes=4)[0]
    first = RandomTopological(seed=7).call(network, large_random_dag)
    second = RandomTopological(seed=7).call(network, large_random_dag)
    assert first == second
    assert first != RandomTopological(seed=8).call(network, large_random_dag)


def test_unseeded_random_topological_varies(unequal_paths_graph):
    network = gen_random_networks(num=1, num_nodes=3)[0]
    random.seed(0)
    orders = {
        tuple(RandomTopological().call(network, unequal_paths_graph)) for _ in range(50)
    }
    assert len(orders) > 1
    for order in orders:
        assert _is_topological(list(order), unequal_paths_graph)


@pytest.mark.parametrize(
    "insert_name", ["EFT_Insert", "EFT_Insert_CP", "EFT_Append"]
)
def test_random_order_parametric_schedulers_produce_valid_schedules(
    large_random_dag, insert_name
):
    network = gen_random_networks(num=1, num_nodes=4)[0]
    scheduler = ParametricScheduler(
        initial_priority=RandomTopological(seed=3), insert_task=insert_funcs[insert_name]
    )
    schedule = scheduler.schedule(network, large_random_dag)
    scheduled = {t.name: t for _, tasks in schedule.items() for t in tasks}
    assert set(scheduled) == {t.name for t in large_random_dag.tasks}
    for dep in large_random_dag.dependencies:
        assert scheduled[dep.source].end <= scheduled[dep.target].start + 1e-9


def test_random_insert_honors_node_constraints(unequal_paths_graph):
    network = gen_random_networks(num=1, num_nodes=4)[0]
    node_names = sorted(node.name for node in network.nodes)
    constraints = {"D": {node_names[0]}, "A": {node_names[1], node_names[2]}}
    scheduler = ParametricScheduler(
        initial_priority=RandomTopological(seed=1), insert_task=RandomInsert(seed=1)
    )
    for seed in range(20):
        scheduler = ParametricScheduler(
            initial_priority=RandomTopological(seed=seed), insert_task=RandomInsert(seed=seed)
        )
        schedule = scheduler.schedule(
            network, unequal_paths_graph, node_constraints=constraints
        )
        scheduled = {t.name: t for _, tasks in schedule.items() for t in tasks}
        assert scheduled["D"].node == node_names[0]
        assert scheduled["A"].node in constraints["A"]
        for dep in unequal_paths_graph.dependencies:
            assert scheduled[dep.source].end <= scheduled[dep.target].start + 1e-9
