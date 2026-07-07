"""Tests that placement constraints flow through the online simulation stack.

node_constraints passed to Environment / StochasticEnvironment must be honored by the
initial schedule and by every policy re-plan (static, reschedule, inspirit), in both the
deterministic and stochastic regimes.
"""

import pytest

from saga import Network, TaskGraph
from saga.stochastic import StochasticNetwork, StochasticTaskGraph
from saga.schedulers.online.environment import Environment, StochasticEnvironment
from saga.schedulers.online.policy import ReschedulePolicy, InspiritPolicy
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    UpwardRanking,
    GreedyInsert,
    GreedyInsertCompareFuncs,
)

CONSTRAINTS = {"a": {"v1"}, "d": {"v2"}}


def _scheduler():
    return ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(compare=GreedyInsertCompareFuncs.EFT),
    )


def _policies():
    return {
        "static": None,
        "reschedule": ReschedulePolicy(),
        "inspirit": InspiritPolicy(delta_ready=1, dec_step=1, s_inc=1, s_dec=1),
    }


def _placement(schedule):
    return {t.name: t.node for tasks in schedule.mapping.values() for t in tasks}


@pytest.fixture
def det_instance():
    net = Network.create(
        nodes=[("v1", 1.0), ("v2", 1.0), ("v3", 10.0)],
        edges=[
            ("v1", "v2", 1.0), ("v1", "v3", 1.0), ("v2", "v3", 1.0),
            ("v1", "v1", 1e9), ("v2", "v2", 1e9), ("v3", "v3", 1e9),
        ],
    )
    tg = TaskGraph.create(
        tasks=[("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)],
        dependencies=[("a", "b", 1.0), ("a", "c", 1.0), ("b", "d", 1.0), ("c", "d", 1.0)],
    )
    return net, tg


@pytest.fixture
def stoch_instance():
    net = StochasticNetwork.create(
        nodes=[("v1", 1.0), ("v2", 1.0), ("v3", 10.0)],
        edges=[("v1", "v2", 1.0), ("v1", "v3", 1.0), ("v2", "v3", 1.0)],
    )
    tg = StochasticTaskGraph.create(
        tasks=[("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)],
        dependencies=[("a", "b", 1.0), ("a", "c", 1.0), ("b", "d", 1.0), ("c", "d", 1.0)],
    )
    return net, tg


@pytest.mark.parametrize("policy_name", ["static", "reschedule", "inspirit"])
def test_deterministic_online_honors_constraints(det_instance, policy_name):
    net, tg = det_instance
    policy = _policies()[policy_name]
    final = Environment(
        net, tg, scheduler=_scheduler(), policy=policy, node_constraints=CONSTRAINTS
    ).run()
    placement = _placement(final)
    assert placement["a"] == "v1"
    assert placement["d"] == "v2"


@pytest.mark.parametrize("policy_name", ["static", "reschedule", "inspirit"])
def test_stochastic_online_honors_constraints(stoch_instance, policy_name):
    net, tg = stoch_instance
    policy = _policies()[policy_name]
    final = StochasticEnvironment(
        net, tg, scheduler=_scheduler(), estimate=lambda rv: rv.mean(),
        policy=policy, seed=0, node_constraints=CONSTRAINTS,
    ).run()
    placement = _placement(final)
    assert placement["a"] == "v1"
    assert placement["d"] == "v2"
