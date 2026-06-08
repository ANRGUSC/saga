"""Tests for the StochasticEnvironment / OnlineHEFT against the reference OnlineParametricScheduler."""
import numpy as np
import pytest
import networkx as nx

from saga.stochastic import StochasticNetwork, StochasticTaskGraph
from saga.utils.random_variable import UniformRandomVariable
from saga.schedulers.online.online_algorithms.online_heft import OnlineHEFTEnvironment
from saga.schedulers.parametric.online import OnlineParametricScheduler
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import GreedyInsert, GreedyInsertCompareFuncs, UpwardRanking

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scheduler() -> ParametricScheduler:
    return ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(
            append_only=False,
            compare=GreedyInsertCompareFuncs.EFT,
            critical_path=False,
        ),
    )


# ---------------------------------------------------------------------------
# Fixtures — use real random variables with variance
# ---------------------------------------------------------------------------

@pytest.fixture()
def chain_instance():
    """3-task chain (t1→t2→t3) on 2 nodes with stochastic weights."""
    network = nx.Graph()
    network.add_node("v1", weight=UniformRandomVariable(1, 3))
    network.add_node("v2", weight=UniformRandomVariable(2, 4))
    network.add_edge("v1", "v2", weight=UniformRandomVariable(1, 2))

    task_graph = nx.DiGraph()
    task_graph.add_node("t1", weight=UniformRandomVariable(2, 6))
    task_graph.add_node("t2", weight=UniformRandomVariable(3, 7))
    task_graph.add_node("t3", weight=UniformRandomVariable(1, 5))
    task_graph.add_edge("t1", "t2", weight=UniformRandomVariable(1, 3))
    task_graph.add_edge("t2", "t3", weight=UniformRandomVariable(1, 2))

    return StochasticNetwork.from_nx(network), StochasticTaskGraph.from_nx(task_graph)


@pytest.fixture()
def diamond_instance():
    """Diamond DAG (t1→t2,t3→t4) on 2 nodes with stochastic weights."""
    network = nx.Graph()
    network.add_node("v1", weight=UniformRandomVariable(1, 4))
    network.add_node("v2", weight=UniformRandomVariable(2, 5))
    network.add_edge("v1", "v2", weight=UniformRandomVariable(1, 3))

    task_graph = nx.DiGraph()
    task_graph.add_node("t1", weight=UniformRandomVariable(2, 5))
    task_graph.add_node("t2", weight=UniformRandomVariable(3, 6))
    task_graph.add_node("t3", weight=UniformRandomVariable(2, 4))
    task_graph.add_node("t4", weight=UniformRandomVariable(4, 8))
    task_graph.add_edge("t1", "t2", weight=UniformRandomVariable(1, 3))
    task_graph.add_edge("t1", "t3", weight=UniformRandomVariable(2, 4))
    task_graph.add_edge("t2", "t4", weight=UniformRandomVariable(1, 2))
    task_graph.add_edge("t3", "t4", weight=UniformRandomVariable(1, 3))

    return StochasticNetwork.from_nx(network), StochasticTaskGraph.from_nx(task_graph)


# ---------------------------------------------------------------------------
# Smoke tests — OnlineHEFT runs without error
# ---------------------------------------------------------------------------

class TestOnlineHEFTSmoke:
    def test_chain_completes(self, chain_instance):
        nw, tg = chain_instance
        schedule = OnlineHEFTEnvironment(nw, tg, seed=SEED).run()
        assert schedule is not None

    def test_diamond_completes(self, diamond_instance):
        nw, tg = diamond_instance
        schedule = OnlineHEFTEnvironment(nw, tg, seed=SEED).run()
        assert schedule is not None

    def test_makespan_positive(self, chain_instance):
        nw, tg = chain_instance
        schedule = OnlineHEFTEnvironment(nw, tg, seed=SEED).run()
        assert schedule.makespan > 0

    def test_all_tasks_scheduled(self, diamond_instance):
        nw, tg = diamond_instance
        schedule = OnlineHEFTEnvironment(nw, tg, seed=SEED).run()
        scheduled_names = {t.name for tasks in schedule.mapping.values() for t in tasks}
        task_names = {task.name for task in tg.tasks}
        assert task_names.issubset(scheduled_names)

    def test_run_is_idempotent(self, chain_instance):
        """Same seed must produce the same makespan across multiple runs."""
        nw, tg = chain_instance
        env = OnlineHEFTEnvironment(nw, tg, seed=SEED)
        ms1 = env.run().makespan
        ms2 = env.run().makespan
        assert ms1 == ms2


# ---------------------------------------------------------------------------
# Correctness — dependency order respected
# ---------------------------------------------------------------------------

class TestDependencyOrder:
    def _end(self, schedule, name: str) -> float:
        for tasks in schedule.mapping.values():
            for t in tasks:
                if t.name == name:
                    return t.end
        raise KeyError(name)

    def _start(self, schedule, name: str) -> float:
        for tasks in schedule.mapping.values():
            for t in tasks:
                if t.name == name:
                    return t.start
        raise KeyError(name)

    def test_chain_order(self, chain_instance):
        nw, tg = chain_instance
        schedule = OnlineHEFTEnvironment(nw, tg, seed=SEED).run()
        assert self._end(schedule, "t1") <= self._start(schedule, "t2")
        assert self._end(schedule, "t2") <= self._start(schedule, "t3")

    def test_diamond_order(self, diamond_instance):
        nw, tg = diamond_instance
        schedule = OnlineHEFTEnvironment(nw, tg, seed=SEED).run()
        assert self._end(schedule, "t1") <= self._start(schedule, "t2")
        assert self._end(schedule, "t1") <= self._start(schedule, "t3")
        assert self._end(schedule, "t2") <= self._start(schedule, "t4")
        assert self._end(schedule, "t3") <= self._start(schedule, "t4")


# ---------------------------------------------------------------------------
# Parity with reference OnlineParametricScheduler
# ---------------------------------------------------------------------------

class TestParityWithReference:
    """Both implementations must produce the same makespan when given
    identical random samples.  OnlineHEFTEnvironment seeds numpy in reset()
    via the seed parameter; we seed identically before calling the reference
    so both draw the same actual_task_graph / actual_network samples.
    """

    def _env_makespan(self, nw, tg) -> float:
        return OnlineHEFTEnvironment(nw, tg, seed=SEED).run().makespan

    def _ref_makespan(self, nw, tg) -> float:
        np.random.seed(SEED)
        return OnlineParametricScheduler(
            scheduler=_make_scheduler(),
            estimate=lambda rv: rv.mean(),
        ).schedule(nw, tg).makespan

    def test_chain_makespan_matches_reference(self, chain_instance):
        nw, tg = chain_instance
        assert self._env_makespan(nw, tg) == self._ref_makespan(nw, tg)

    def test_diamond_makespan_matches_reference(self, diamond_instance):
        nw, tg = diamond_instance
        assert self._env_makespan(nw, tg) == self._ref_makespan(nw, tg)
