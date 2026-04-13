"""Tests for ``saga.execution``.

These tests exercise the Python-decorator notation, the SagaGraph → TaskGraph
conversion, the Makeflow JSON emitter, the worker wrapper, and the
dagprofiler profile adapter.  They do NOT require CCTools / Work Queue to be
installed, so they are safe to run in plain CI.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from saga import Network, Schedule
from saga.execution import SagaGraph, saga_task
from saga.execution.chameleon import ChameleonNode, build_network
from saga.execution.makeflow import emit_makeflow
from saga.execution.profile_adapter import profile_to_task_graph
from saga.execution.workqueue import feature_for
from saga.schedulers import HeftScheduler


# ---------------------------------------------------------------------------
# Reusable test graph.
# ---------------------------------------------------------------------------


@saga_task(
    inputs=[],
    outputs=["a"],
    cost=lambda cfg: cfg["N"],
    output_sizes={"a": lambda cfg: cfg["N"] * 8},
)
def produce(cfg):
    return {"a": [cfg["N"], cfg["N"] + 1]}


@saga_task(
    inputs=["a"],
    outputs=["b"],
    cost=lambda cfg: 2 * cfg["N"],
    output_sizes={"b": lambda cfg: 16},
)
def double(a, cfg):
    return {"b": [x * 2 for x in a]}


@saga_task(
    inputs=["a"],
    outputs=["c"],
    cost=lambda cfg: 3 * cfg["N"],
    output_sizes={"c": lambda cfg: 16},
)
def triple(a, cfg):
    return {"c": [x * 3 for x in a]}


@saga_task(
    inputs=["b", "c"],
    outputs=["result"],
    cost=lambda cfg: cfg["N"],
    output_sizes={"result": lambda cfg: 8},
)
def combine(b, c, cfg):
    return {"result": sum(b) + sum(c)}


def _graph() -> SagaGraph:
    return SagaGraph(
        tasks=[produce, double, triple, combine],
        config={"N": 4},
    )


# ---------------------------------------------------------------------------
# Graph construction.
# ---------------------------------------------------------------------------


def test_decorator_yields_saga_task():
    assert produce.name == "produce"
    assert produce.inputs == ()
    assert produce.outputs == ("a",)
    # The decorator preserves the function so we can still call it.
    out = produce.fn(cfg={"N": 5})
    assert out == {"a": [5, 6]}


def test_edges_are_inferred_from_input_names():
    g = _graph()
    edges = {(s, d) for s, d, _ in g.edges}
    assert edges == {
        ("produce", "double"),
        ("produce", "triple"),
        ("double", "combine"),
        ("triple", "combine"),
    }


def test_missing_producer_raises():
    @saga_task(inputs=["missing"], outputs=["x"])
    def consumer(missing, cfg):
        return {"x": missing}

    with pytest.raises(ValueError, match="no task produces it"):
        SagaGraph([consumer])


def test_duplicate_output_name_raises():
    @saga_task(inputs=[], outputs=["dup"])
    def p1(cfg):
        return {"dup": 1}

    @saga_task(inputs=[], outputs=["dup"])
    def p2(cfg):
        return {"dup": 2}

    with pytest.raises(ValueError, match="rename one of them"):
        SagaGraph([p1, p2])


def test_to_task_graph_respects_cost_and_size_models():
    g = _graph()
    tg = g.to_task_graph()

    by_name = {t.name: t for t in tg.tasks}
    # Non-synthetic tasks should have the declared costs.
    assert by_name["produce"].cost == pytest.approx(4.0)
    assert by_name["double"].cost == pytest.approx(8.0)

    # produce->double edge carries an 'a' payload of 32 bytes.
    by_edge = {(e.source, e.target): e for e in tg.dependencies}
    assert by_edge[("produce", "double")].size == pytest.approx(32.0)


def test_overrides_take_precedence():
    g = _graph()
    tg = g.to_task_graph(
        node_cost_overrides={"produce": 999.0},
        edge_size_overrides={("produce", "double"): 7.0},
    )
    by_name = {t.name: t for t in tg.tasks}
    by_edge = {(e.source, e.target): e for e in tg.dependencies}
    assert by_name["produce"].cost == pytest.approx(999.0)
    assert by_edge[("produce", "double")].size == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Chameleon network helper.
# ---------------------------------------------------------------------------


def test_build_network_is_fully_connected():
    nodes = [
        ChameleonNode("a", speed=1.0),
        ChameleonNode("b", speed=2.0),
        ChameleonNode("c", speed=3.0),
    ]
    net = build_network(nodes, default_bandwidth=100.0)
    assert isinstance(net, Network)
    names = {n.name for n in net.nodes}
    assert names == {"a", "b", "c"}
    # Self-loops get infinite bandwidth; cross edges use the default.
    ab = net.get_edge("a", "b")
    assert ab.speed == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Makeflow emitter.
# ---------------------------------------------------------------------------


def _schedule(graph: SagaGraph) -> tuple[Network, Schedule]:
    net = build_network(
        [ChameleonNode("alpha", 2.0), ChameleonNode("beta", 1.0)],
        default_bandwidth=1.0,
    )
    tg = graph.to_task_graph()
    sched = HeftScheduler().schedule(net, tg)
    return net, sched


def test_emit_makeflow_writes_valid_json(tmp_path: Path):
    g = _graph()
    _, sched = _schedule(g)
    artifacts = emit_makeflow(g, sched, tmp_path, task_module="tests.test_execution")

    assert artifacts.makeflow_json.exists()
    workflow = json.loads(artifacts.makeflow_json.read_text())

    # Basic schema sanity.
    assert "rules" in workflow and isinstance(workflow["rules"], list)
    assert "categories" in workflow
    assert len(workflow["rules"]) == len(g.tasks)
    assert set(workflow["categories"]) == {"node_alpha", "node_beta"}

    # Each rule should pin to one of the two categories.
    for rule in workflow["rules"]:
        assert rule["category"] in {"node_alpha", "node_beta"}
        assert "command" in rule
        assert "saga.execution._wrapper" in rule["command"]

    # The config pickle was staged and listed as an input of every rule.
    assert (tmp_path / "config.pkl").exists()
    for rule in workflow["rules"]:
        assert "config.pkl" in rule["inputs"]


def test_emit_makeflow_requires_task_module(tmp_path: Path):
    g = _graph()
    _, sched = _schedule(g)
    with pytest.raises(ValueError, match="task_module"):
        emit_makeflow(g, sched, tmp_path)


# ---------------------------------------------------------------------------
# Wrapper subprocess — prove it really executes a task.
# ---------------------------------------------------------------------------


def test_wrapper_invokes_task_end_to_end(tmp_path: Path):
    # Write a tiny module that defines a @saga_task-decorated function.
    mod_path = tmp_path / "mini_mod.py"
    mod_path.write_text(
        textwrap.dedent("""
            from saga.execution import saga_task

            @saga_task(inputs=["x"], outputs=["y"], cost=lambda cfg: 1.0)
            def scale(x, cfg):
                return {"y": [v * cfg["k"] for v in x]}
        """).strip()
    )

    # Stage the config and input.
    (tmp_path / "config.pkl").write_bytes(pickle.dumps({"k": 3}))
    (tmp_path / "x.pkl").write_bytes(pickle.dumps([1, 2, 5]))

    # Invoke the wrapper as the workers would.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "saga.execution._wrapper",
            "--task-module",
            "mini_mod",
            "--task-attr",
            "scale",
            "--config",
            "config.pkl",
            "--input",
            "x=x.pkl",
            "--output",
            "y=y.pkl",
        ],
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), **_env_passthrough()},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"wrapper failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    with (tmp_path / "y.pkl").open("rb") as fh:
        out = pickle.load(fh)
    assert out == [3, 6, 15]


def _env_passthrough() -> dict[str, str]:
    import os

    keys = {"PATH", "HOME", "LANG", "LC_ALL", "VIRTUAL_ENV", "CONDA_PREFIX"}
    return {k: v for k, v in os.environ.items() if k in keys}


# ---------------------------------------------------------------------------
# dagprofiler profile adapter.
# ---------------------------------------------------------------------------


def test_profile_adapter_uses_node_weights_and_edge_weights(tmp_path: Path):
    profile = {
        "dag_structure": {
            "nodes": ["a", "b", "c"],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
            ],
        },
        "node_weights": {"a": 100, "b": 200, "c": 300},
        "edge_weights": {"a->b": 800, "b->c": 1600},  # bits
        "execution_metrics": [
            {"task": "a", "compute_time_ms": 5.0},
            {"task": "b", "compute_time_ms": 10.0},
            {"task": "c", "compute_time_ms": 15.0},
        ],
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))

    tg = profile_to_task_graph(path, edge_units="bytes")

    by_name = {t.name: t for t in tg.tasks}
    assert by_name["a"].cost == pytest.approx(100.0)
    assert by_name["b"].cost == pytest.approx(200.0)

    by_edge = {(e.source, e.target): e for e in tg.dependencies}
    # 800 bits / 8 = 100 bytes
    assert by_edge[("a", "b")].size == pytest.approx(100.0)
    assert by_edge[("b", "c")].size == pytest.approx(200.0)


def test_profile_adapter_can_use_measured_time(tmp_path: Path):
    profile = {
        "dag_structure": {
            "nodes": ["a", "b"],
            "edges": [{"source": "a", "target": "b"}],
        },
        "node_weights": {"a": 999, "b": 999},
        "edge_weights": {"a->b": 80},
        "execution_metrics": [
            {"task": "a", "compute_time_ms": 5.25},
            {"task": "b", "compute_time_ms": 7.75},
        ],
    }
    tg = profile_to_task_graph(profile, use_measured_time=True)
    by_name = {t.name: t for t in tg.tasks}
    assert by_name["a"].cost == pytest.approx(5.25)
    assert by_name["b"].cost == pytest.approx(7.75)


# ---------------------------------------------------------------------------
# Feature-string helper (used by both workqueue.py and provision_workers.sh).
# ---------------------------------------------------------------------------


def test_feature_for_sanitises_non_alphanumerics():
    assert feature_for("saga-worker-0") == "saga-node-saga_worker_0"
    assert feature_for("a.b-c_d") == "saga-node-a_b_c_d"
