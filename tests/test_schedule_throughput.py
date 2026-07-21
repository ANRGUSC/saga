"""Tests for Schedule.throughput."""

import pytest

from saga import Network, ScheduledTask, Schedule, TaskGraph


def _schedule(task_cost: float, node_speed: float) -> Schedule:
    network = Network.create(nodes=[("A", node_speed)], edges=[])
    task_graph = TaskGraph.create(tasks=[("t1", task_cost)], dependencies=[])
    return Schedule(task_graph, network)


def test_throughput_is_inverse_of_bottleneck():
    schedule = _schedule(task_cost=4.0, node_speed=2.0)
    schedule.add_task(ScheduledTask(node="A", name="t1", start=0.0, end=2.0))
    assert schedule.throughput == pytest.approx(0.5)


def test_throughput_rejects_empty_schedule():
    with pytest.raises(ValueError):
        _schedule(task_cost=1.0, node_speed=1.0).throughput


def test_throughput_rejects_zero_bottleneck():
    # Every scheduled task takes zero time, so there is no meaningful bottleneck.
    schedule = _schedule(task_cost=0.0, node_speed=1.0)
    schedule.add_task(ScheduledTask(node="A", name="t1", start=0.0, end=0.0))
    with pytest.raises(ValueError):
        schedule.throughput
