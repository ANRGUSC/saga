"""Tests for task duplication: the multi-copy Schedule model and the HEFT/CPOP hooks."""

import math

import pytest

from saga import Network, Schedule, ScheduledTask, TaskGraph
from saga.schedulers.cpop import CpopScheduler
from saga.schedulers.heft import HeftScheduler


def _comm_heavy_fork():
    """Fast nodes, slow links, one source feeding three children with large data."""
    network = Network.create(
        nodes=[("A", 10.0), ("B", 10.0), ("C", 10.0)],
        edges=[("A", "B", 1.0), ("A", "C", 1.0), ("B", "C", 1.0)],
    )
    task_graph = TaskGraph.create(
        tasks=[("S", 1.0), ("c1", 1.0), ("c2", 1.0), ("c3", 1.0)],
        dependencies=[("S", "c1", 50.0), ("S", "c2", 50.0), ("S", "c3", 50.0)],
    )
    return network, task_graph


def _empty_schedule():
    network = Network.create(nodes=[("A", 1.0), ("B", 1.0)], edges=[("A", "B", 1.0)])
    task_graph = TaskGraph.create(
        tasks=[("p", 1.0), ("t", 1.0)], dependencies=[("p", "t", 1.0)]
    )
    return Schedule(task_graph, network), network, task_graph


# --- model ---------------------------------------------------------------


def test_add_task_allows_same_name_on_different_nodes():
    schedule, _, _ = _empty_schedule()
    schedule.add_task(ScheduledTask(node="A", name="p", start=0.0, end=1.0))
    schedule.add_task(ScheduledTask(node="B", name="p", start=0.0, end=1.0))
    assert {t.node for t in schedule.get_scheduled_tasks("p")} == {"A", "B"}


def test_add_task_rejects_same_name_on_same_node():
    schedule, _, _ = _empty_schedule()
    schedule.add_task(ScheduledTask(node="A", name="p", start=0.0, end=1.0))
    with pytest.raises(ValueError):
        schedule.add_task(ScheduledTask(node="A", name="p", start=2.0, end=3.0))


def test_get_scheduled_task_returns_primary_copy():
    schedule, _, _ = _empty_schedule()
    schedule.add_task(ScheduledTask(node="A", name="p", start=0.0, end=1.0))
    schedule.add_task(ScheduledTask(node="B", name="p", start=0.0, end=1.0))
    assert schedule.get_scheduled_task("p").node == "A"  # first placed
    assert len(schedule.get_scheduled_tasks("p")) == 2


def test_remove_task_removes_all_copies_and_clears_duplication():
    schedule, _, task_graph = _empty_schedule()
    schedule.add_task(ScheduledTask(node="A", name="p", start=0.0, end=1.0))
    schedule.add_task(ScheduledTask(node="B", name="p", start=0.0, end=1.0))
    schedule.remove_task("p")
    assert not schedule.is_scheduled("p")
    assert all(t.name != "p" for tasks in schedule.mapping.values() for t in tasks)
    # throughput was undefined while p was duplicated; removing it clears that.
    schedule.add_task(ScheduledTask(node="A", name="p", start=0.0, end=1.0))
    schedule.add_task(ScheduledTask(node="B", name="t", start=1.0, end=2.0))
    assert schedule.throughput > 0


def test_get_earliest_start_time_uses_closest_parent_copy():
    # Parent p on A (end 1.0) and on B (end 1.0); child t on B reads the local copy
    # (self-loop = infinite speed = zero transfer) instead of the far A copy.
    network = Network.create(nodes=[("A", 1.0), ("B", 1.0)], edges=[("A", "B", 0.1)])
    task_graph = TaskGraph.create(
        tasks=[("p", 1.0), ("t", 1.0)], dependencies=[("p", "t", 100.0)]
    )
    schedule = Schedule(task_graph, network)
    schedule.add_task(ScheduledTask(node="A", name="p", start=0.0, end=1.0))
    schedule.add_task(ScheduledTask(node="B", name="p", start=0.0, end=1.0))
    # From A: 1.0 + 100/0.1 = 1001. From B (local): 1.0 + 0 = 1.0. Best copy wins.
    assert schedule.get_earliest_start_time(
        "t", "B", append_only=True
    ) == pytest.approx(1.0)


def test_throughput_and_bottleneck_raise_under_duplication():
    schedule, _, _ = _empty_schedule()
    schedule.add_task(ScheduledTask(node="A", name="p", start=0.0, end=1.0))
    schedule.add_task(ScheduledTask(node="B", name="p", start=0.0, end=1.0))
    with pytest.raises(ValueError):
        schedule.throughput
    with pytest.raises(ValueError):
        schedule.bottleneck_if_added(
            ScheduledTask(node="A", name="t", start=1.0, end=2.0)
        )


# --- schedulers ----------------------------------------------------------


def test_heft_default_does_not_duplicate():
    network, task_graph = _comm_heavy_fork()
    schedule = HeftScheduler().schedule(network, task_graph)
    for name in (t.name for t in task_graph.tasks):
        assert len(schedule.get_scheduled_tasks(name)) == 1


def test_heft_duplication_reduces_makespan_on_comm_heavy_fork():
    network, task_graph = _comm_heavy_fork()
    base = HeftScheduler(duplication_factor=1).schedule(network, task_graph)
    dup = HeftScheduler(duplication_factor=3).schedule(network, task_graph)
    assert dup.makespan < base.makespan
    assert len(dup.get_scheduled_tasks("S")) > 1  # source was duplicated


def test_cpop_duplicates_non_critical_comm_heavy_task():
    # H is a long critical path; A is comm-heavy but not on it, so it may duplicate.
    network = Network.create(
        nodes=[("A", 10.0), ("B", 10.0), ("C", 10.0)],
        edges=[("A", "B", 1.0), ("A", "C", 1.0), ("B", "C", 1.0)],
    )
    task_graph = TaskGraph.create(
        tasks=[
            ("R", 1.0),
            ("H", 400.0),
            ("A", 1.0),
            ("c1", 1.0),
            ("c2", 1.0),
            ("SINK", 1.0),
        ],
        dependencies=[
            ("R", "H", 0.1),
            ("R", "A", 0.1),
            ("A", "c1", 5.0),
            ("A", "c2", 5.0),
            ("H", "SINK", 0.1),
            ("c1", "SINK", 0.1),
            ("c2", "SINK", 0.1),
        ],
    )
    schedule = CpopScheduler(duplication_factor=2).schedule(network, task_graph)
    assert len(schedule.get_scheduled_tasks("A")) == 2


@pytest.mark.parametrize(
    "scheduler",
    [HeftScheduler(duplication_factor=3), CpopScheduler(duplication_factor=3)],
)
def test_duplicated_schedule_places_every_task(scheduler):
    network, task_graph = _comm_heavy_fork()
    schedule = scheduler.schedule(network, task_graph)
    placed = {t.name for tasks in schedule.mapping.values() for t in tasks}
    assert placed == {t.name for t in task_graph.tasks}
    assert math.isfinite(schedule.makespan)
