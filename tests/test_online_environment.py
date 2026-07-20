"""Tests for the online simulation loop and its per-step state bookkeeping."""

import pytest

from saga.schedulers.online.algorithms.fifo import FIFOEnvironment
from saga.schedulers.online.algorithms.online_heft import OnlineHEFTEnvironment
from saga.stochastic import StochasticNetwork, StochasticTaskGraph
from saga.utils.random_graphs import get_chain_dag, get_diamond_dag, get_network
from saga.utils.random_variable import RandomVariable


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


@pytest.mark.parametrize("task_graph", [get_diamond_dag(), get_chain_dag()])
def test_frontier_environment_runs_and_records_history(task_graph):
    env = FIFOEnvironment(get_network(num_nodes=4), task_graph)
    schedule = env.run()

    expected = {task.name for task in task_graph.tasks}
    placed = {t.name for tasks in schedule.mapping.values() for t in tasks}
    assert placed == expected
    assert env.history, "the run recorded no steps"
    assert env.history[-1].finished_tasks == frozenset(expected)


def test_stochastic_environment_runs_and_records_history():
    network, task_graph = _stochastic_chain()
    env = OnlineHEFTEnvironment(network, task_graph)
    env.run()

    expected = {task.name for task in task_graph.tasks}
    assert env.history
    assert env.history[-1].finished_tasks == frozenset(expected)


@pytest.mark.parametrize(
    "make_env",
    [
        lambda: FIFOEnvironment(get_network(num_nodes=4), get_diamond_dag()),
        lambda: OnlineHEFTEnvironment(*_stochastic_chain()),
    ],
    ids=["frontier", "stochastic"],
)
def test_task_state_sets_partition_the_task_graph(make_env):
    """finished, ready and unready stay disjoint and never lose a task."""
    env = make_env()
    env.run()
    all_names = {task.name for task in env.task_graph.tasks}

    for record in env.history:
        assert record.finished_tasks.isdisjoint(record.unready_tasks)
        assert record.ready_tasks.isdisjoint(record.unready_tasks)
        assert record.finished_tasks <= all_names
        assert record.unready_tasks <= all_names


@pytest.mark.parametrize(
    "make_env",
    [
        lambda: FIFOEnvironment(get_network(num_nodes=4), get_chain_dag()),
        lambda: OnlineHEFTEnvironment(*_stochastic_chain()),
    ],
    ids=["frontier", "stochastic"],
)
def test_unready_tasks_shrink_as_tasks_finish(make_env):
    """A task never returns to unready once it is finished, and the set drains."""
    env = make_env()
    env.run()

    for record in env.history:
        assert record.finished_tasks.isdisjoint(record.unready_tasks)
    assert env.history[-1].unready_tasks == frozenset()


def test_history_time_is_non_decreasing():
    env = FIFOEnvironment(get_network(num_nodes=4), get_diamond_dag())
    env.run()
    times = [record.time for record in env.history]
    assert times == sorted(times)
