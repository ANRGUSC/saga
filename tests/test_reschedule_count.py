"""Tests for Environment.reschedule_count."""

from typing import Optional

from saga import Schedule
from saga.schedulers.heft import HeftScheduler
from saga.schedulers.online.algorithms.fifo import FIFOEnvironment
from saga.schedulers.online.algorithms.online_heft import OnlineHEFTEnvironment
from saga.schedulers.online.environment import Environment
from saga.schedulers.online.policy import OnlinePolicy
from saga.stochastic import StochasticNetwork, StochasticTaskGraph
from saga.utils.random_graphs import get_diamond_dag, get_network
from saga.utils.random_variable import RandomVariable


class _NeverReschedule(OnlinePolicy):
    """A policy that inspects the environment but never revises the schedule."""

    def update(self, environment: Environment) -> Optional[Schedule]:
        return None


def _stochastic_chain():
    def rv():
        return RandomVariable(samples=[1.0, 2.0])

    network = StochasticNetwork.create(
        nodes=[("A", rv()), ("B", rv())], edges=[("A", "B", rv())]
    )
    task_graph = StochasticTaskGraph.create(
        tasks=[("t1", rv()), ("t2", rv()), ("t3", rv())],
        dependencies=[("t1", "t2", rv()), ("t2", "t3", rv())],
    )
    return network, task_graph


def test_reschedule_count_starts_at_zero():
    env = OnlineHEFTEnvironment(*_stochastic_chain())
    assert env.reschedule_count == 0


def test_reschedule_count_resets_each_run():
    env = OnlineHEFTEnvironment(*_stochastic_chain())
    env.run()
    first = env.reschedule_count
    assert first > 0
    env.run()
    assert env.reschedule_count == first  # reset, not accumulated across runs


def test_frontier_environment_reset_zeroes_reschedule_count():
    env = FIFOEnvironment(get_network(num_nodes=4), get_diamond_dag())
    env.reschedule_count = 7
    env.reset()
    assert env.reschedule_count == 0


def test_never_rescheduling_policy_leaves_count_zero():
    # The step loop itself must not touch reschedule_count; only a real reschedule
    # inside a policy increments it.
    env = Environment(
        get_network(num_nodes=4),
        get_diamond_dag(),
        scheduler=HeftScheduler(),
        policy=_NeverReschedule(),
    )
    env.run()
    assert env.history, "the run recorded no steps"
    assert env.reschedule_count == 0


def test_always_rescheduling_policy_counts_every_reschedule():
    # OnlineHEFT uses ReschedulePolicy, which reschedules on every step.
    env = OnlineHEFTEnvironment(*_stochastic_chain())
    env.run()
    assert env.reschedule_count == len(env.history) > 0
