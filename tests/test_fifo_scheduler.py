"""Tests for FIFOEnvironment and FrontierFillPolicy.

Design: pure FIFO scheduler that builds the schedule incrementally.
  - Root tasks (no predecessors) are bootstrapped into the schedule during reset()
    so the simulation has events to advance to.
  - As each task finishes, its newly-ready successors are enqueued on the frontier
    (a min-heap keyed on arrival time).
  - On every step the FrontierFillPolicy pops the oldest-ready task and inserts
    it into the schedule via GreedyInsert(EST).
"""

import heapq
import pytest

from saga import Network, Schedule, ScheduledTask, TaskGraph
from saga.schedulers.online.policy import FrontierFillPolicy
from saga.schedulers.online.environment import FrontierEnvironment
from saga.schedulers.online.algorithms.fifo import FIFOEnvironment
from saga.schedulers.parametric.components import GreedyInsert, GreedyInsertCompareFuncs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_node_network():
    return Network.create(
        nodes=[("v1", 1.0), ("v2", 1.0)],
        edges=[("v1", "v2", 1.0)],
    )


@pytest.fixture
def chain_task_graph():
    """t1 → t2 → t3"""
    return TaskGraph.create(
        tasks=[("t1", 2.0), ("t2", 3.0), ("t3", 1.0)],
        dependencies=[("t1", "t2", 0.0), ("t2", "t3", 0.0)],
    )


@pytest.fixture
def diamond_task_graph():
    """t1 → t2, t1 → t3, t2 → t4, t3 → t4"""
    return TaskGraph.create(
        tasks=[("t1", 1.0), ("t2", 2.0), ("t3", 2.0), ("t4", 1.0)],
        dependencies=[
            ("t1", "t2", 0.0),
            ("t1", "t3", 0.0),
            ("t2", "t4", 0.0),
            ("t3", "t4", 0.0),
        ],
    )


@pytest.fixture
def wide_task_graph():
    """t1 fans out to t2, t3, t4 simultaneously"""
    return TaskGraph.create(
        tasks=[("t1", 1.0), ("t2", 2.0), ("t3", 1.5), ("t4", 3.0)],
        dependencies=[
            ("t1", "t2", 0.0),
            ("t1", "t3", 0.0),
            ("t1", "t4", 0.0),
        ],
    )


# ---------------------------------------------------------------------------
# Unit — FrontierFillPolicy
# ---------------------------------------------------------------------------

class TestFrontierFillPolicyInit:
    def test_default_construction_creates_insertion_strategy(self):
        """Default FrontierFillPolicy must have a non-None insertion strategy."""
        ctrl = FrontierFillPolicy()
        assert ctrl._insertion_strategy is not None

    def test_custom_strategy_is_preserved(self):
        """Passing an explicit GreedyInsert must keep that exact instance."""
        custom = GreedyInsert(
            append_only=True,
            compare=GreedyInsertCompareFuncs.EFT,
            critical_path=False,
        )
        ctrl = FrontierFillPolicy(insertion_strategy=custom)
        assert ctrl._insertion_strategy is custom

    def test_default_strategy_uses_est(self):
        """Default insertion strategy should use Earliest Start Time (EST)."""
        ctrl = FrontierFillPolicy()
        assert ctrl._insertion_strategy.compare == GreedyInsertCompareFuncs.EST


# ---------------------------------------------------------------------------
# Unit — FIFOEnvironment construction and initial state
# ---------------------------------------------------------------------------

class TestFIFOEnvironmentInit:
    def test_instantiation_does_not_raise(self, two_node_network, chain_task_graph):
        env = FIFOEnvironment(two_node_network, chain_task_graph)
        assert env is not None

    def test_network_stored(self, two_node_network, chain_task_graph):
        env = FIFOEnvironment(two_node_network, chain_task_graph)
        assert env.network is two_node_network

    def test_task_graph_stored(self, two_node_network, chain_task_graph):
        env = FIFOEnvironment(two_node_network, chain_task_graph)
        assert env.task_graph is chain_task_graph

    def test_frontier_empty_before_reset(self, two_node_network, chain_task_graph):
        """Frontier and frontier_set are empty until reset() is called."""
        env = FIFOEnvironment(two_node_network, chain_task_graph)
        assert env.frontier == []
        assert len(env.frontier_set) == 0


# ---------------------------------------------------------------------------
# Unit — frontier state after reset
# ---------------------------------------------------------------------------

class TestFrontierAfterReset:
    """After reset(), root tasks are bootstrapped into the schedule (not the frontier).
    The frontier stays empty until a root task finishes and unblocks successors."""

    def test_frontier_empty_after_reset(self, two_node_network, chain_task_graph):
        env = FIFOEnvironment(two_node_network, chain_task_graph)
        env.reset()
        assert env.frontier == []
        assert len(env.frontier_set) == 0

    def test_root_tasks_in_schedule_after_reset(self, two_node_network, chain_task_graph):
        """Root tasks (those with in-degree 0) must be pre-scheduled so the simulation
        has events to advance to."""
        env = FIFOEnvironment(two_node_network, chain_task_graph)
        env.reset()
        scheduled_names = {
            t.name for tasks in env.schedule.mapping.values() for t in tasks
        }
        root_names = {
            t.name for t in chain_task_graph.tasks
            if chain_task_graph.in_degree(t) == 0
        }
        assert root_names.issubset(scheduled_names)

    def test_schedule_not_empty_after_reset(self, two_node_network, chain_task_graph):
        env = FIFOEnvironment(two_node_network, chain_task_graph)
        env.reset()
        total_tasks = sum(len(v) for v in env.schedule.mapping.values())
        assert total_tasks > 0


# ---------------------------------------------------------------------------
# Unit — frontier state after a step
# ---------------------------------------------------------------------------

class TestFrontierAfterStep:
    def test_successor_dispatched_after_predecessor_finishes(
        self, two_node_network, chain_task_graph
    ):
        """When the root task finishes, its successor becomes ready and is immediately
        dispatched by the controller in the same step (the frontier is populated then
        drained within one step())."""
        env = FIFOEnvironment(two_node_network, chain_task_graph)
        env.reset()
        env.step()  # advances to t1.end; controller inserts t2
        scheduled_names = {
            t.name for tasks in env.schedule.mapping.values() for t in tasks
        }
        assert "t2" in scheduled_names

    def test_frontier_entries_are_name_string_tuples(
        self, two_node_network, chain_task_graph
    ):
        """Every heap entry must be (float, str) so heapq comparisons are safe."""
        env = FIFOEnvironment(two_node_network, chain_task_graph)
        env.reset()
        env.step()
        for entry in env.frontier:
            assert isinstance(entry, tuple) and len(entry) == 2
            arrival_time, task_name = entry
            assert isinstance(arrival_time, (int, float))
            assert isinstance(task_name, str)

    def test_frontier_is_min_heap(self, two_node_network, chain_task_graph):
        """Frontier must satisfy the heap property."""
        env = FIFOEnvironment(two_node_network, chain_task_graph)
        env.reset()
        env.step()
        h = env.frontier
        for i in range(len(h)):
            left, right = 2 * i + 1, 2 * i + 2
            if left < len(h):
                assert h[i] <= h[left]
            if right < len(h):
                assert h[i] <= h[right]

    def test_frontier_set_mirrors_heap(self, two_node_network, chain_task_graph):
        """frontier_set must always contain exactly the names in the frontier heap."""
        env = FIFOEnvironment(two_node_network, chain_task_graph)
        env.reset()
        env.step()
        heap_names = {name for _, name in env.frontier}
        assert heap_names == env.frontier_set

    def test_finished_tasks_not_in_frontier(self, two_node_network, chain_task_graph):
        env = FIFOEnvironment(two_node_network, chain_task_graph)
        env.reset()
        env.step()
        finished_names = {t.name for t in env.finished_tasks}
        frontier_names = {name for _, name in env.frontier}
        assert finished_names.isdisjoint(frontier_names)

    def test_running_tasks_not_in_frontier(self, two_node_network, chain_task_graph):
        env = FIFOEnvironment(two_node_network, chain_task_graph)
        env.reset()
        env.step()
        running_names = {t.name for t in env.running_tasks}
        frontier_names = {name for _, name in env.frontier}
        assert running_names.isdisjoint(frontier_names)


# ---------------------------------------------------------------------------
# Integration — smoke tests
# ---------------------------------------------------------------------------

class TestFIFOSmoke:
    def test_chain_completes(self, two_node_network, chain_task_graph):
        schedule = FIFOEnvironment(two_node_network, chain_task_graph).run()
        assert schedule is not None

    def test_diamond_completes(self, two_node_network, diamond_task_graph):
        schedule = FIFOEnvironment(two_node_network, diamond_task_graph).run()
        assert schedule is not None

    def test_wide_completes(self, two_node_network, wide_task_graph):
        schedule = FIFOEnvironment(two_node_network, wide_task_graph).run()
        assert schedule is not None

    def test_makespan_positive(self, two_node_network, chain_task_graph):
        schedule = FIFOEnvironment(two_node_network, chain_task_graph).run()
        assert schedule.makespan > 0

    def test_run_is_deterministic(self, two_node_network, chain_task_graph):
        """Two identical runs must produce the same makespan."""
        env = FIFOEnvironment(two_node_network, chain_task_graph)
        ms1 = env.run().makespan
        ms2 = env.run().makespan
        assert ms1 == ms2


# ---------------------------------------------------------------------------
# Integration — all tasks scheduled
# ---------------------------------------------------------------------------

class TestAllTasksScheduled:
    def _scheduled_names(self, schedule):
        return {t.name for tasks in schedule.mapping.values() for t in tasks}

    def test_chain_all_scheduled(self, two_node_network, chain_task_graph):
        schedule = FIFOEnvironment(two_node_network, chain_task_graph).run()
        task_names = {t.name for t in chain_task_graph.tasks}
        assert task_names.issubset(self._scheduled_names(schedule))

    def test_diamond_all_scheduled(self, two_node_network, diamond_task_graph):
        schedule = FIFOEnvironment(two_node_network, diamond_task_graph).run()
        task_names = {t.name for t in diamond_task_graph.tasks}
        assert task_names.issubset(self._scheduled_names(schedule))

    def test_wide_all_scheduled(self, two_node_network, wide_task_graph):
        schedule = FIFOEnvironment(two_node_network, wide_task_graph).run()
        task_names = {t.name for t in wide_task_graph.tasks}
        assert task_names.issubset(self._scheduled_names(schedule))


# ---------------------------------------------------------------------------
# Integration — dependency ordering
# ---------------------------------------------------------------------------

class TestDependencyOrdering:
    def _end(self, schedule, name):
        for tasks in schedule.mapping.values():
            for t in tasks:
                if t.name == name:
                    return t.end
        raise KeyError(name)

    def _start(self, schedule, name):
        for tasks in schedule.mapping.values():
            for t in tasks:
                if t.name == name:
                    return t.start
        raise KeyError(name)

    def test_chain_order(self, two_node_network, chain_task_graph):
        schedule = FIFOEnvironment(two_node_network, chain_task_graph).run()
        assert self._end(schedule, "t1") <= self._start(schedule, "t2")
        assert self._end(schedule, "t2") <= self._start(schedule, "t3")

    def test_diamond_t1_before_successors(self, two_node_network, diamond_task_graph):
        schedule = FIFOEnvironment(two_node_network, diamond_task_graph).run()
        assert self._end(schedule, "t1") <= self._start(schedule, "t2")
        assert self._end(schedule, "t1") <= self._start(schedule, "t3")

    def test_diamond_t4_after_predecessors(self, two_node_network, diamond_task_graph):
        """t4 must not start until BOTH t2 and t3 have finished."""
        schedule = FIFOEnvironment(two_node_network, diamond_task_graph).run()
        assert self._end(schedule, "t2") <= self._start(schedule, "t4")
        assert self._end(schedule, "t3") <= self._start(schedule, "t4")

    def test_wide_fanout_order(self, two_node_network, wide_task_graph):
        schedule = FIFOEnvironment(two_node_network, wide_task_graph).run()
        for child in ("t2", "t3", "t4"):
            assert self._end(schedule, "t1") <= self._start(schedule, child)

    def test_all_dependencies_respected(self, two_node_network, chain_task_graph):
        """For every edge u→v in the task graph, end(u) ≤ start(v)."""
        schedule = FIFOEnvironment(two_node_network, chain_task_graph).run()
        for dep in chain_task_graph.dependencies:
            assert self._end(schedule, dep.source) <= self._start(schedule, dep.target), (
                f"Violated: {dep.source} must finish before {dep.target} starts"
            )


# ---------------------------------------------------------------------------
# Integration — FIFO ordering property
# ---------------------------------------------------------------------------

class TestFIFOOrdering:
    def _start(self, schedule, name):
        for tasks in schedule.mapping.values():
            for t in tasks:
                if t.name == name:
                    return t.start
        raise KeyError(name)

    def test_tasks_start_only_after_becoming_ready(self, two_node_network, wide_task_graph):
        """No child task should start before its parent finishes."""
        schedule = FIFOEnvironment(two_node_network, wide_task_graph).run()
        t1_end = None
        for tasks in schedule.mapping.values():
            for t in tasks:
                if t.name == "t1":
                    t1_end = t.end
        assert t1_end is not None
        for child in ("t2", "t3", "t4"):
            assert self._start(schedule, child) >= t1_end - 1e-9

    def test_no_task_scheduled_twice(self, two_node_network, diamond_task_graph):
        """Every task name must appear exactly once in the final schedule."""
        schedule = FIFOEnvironment(two_node_network, diamond_task_graph).run()
        all_names = [t.name for tasks in schedule.mapping.values() for t in tasks]
        assert len(all_names) == len(set(all_names))

    def test_earlier_ready_task_starts_no_later_than_later_ready_task(
        self, two_node_network
    ):
        """With a single node, the task that became ready first (t2, ready at t1.end)
        must start before the task that becomes ready second (t3, ready at t2.end)."""
        network = Network.create(nodes=[("v1", 1.0)], edges=[])
        # Chain: t1 -> t2 -> t3. On one node, t2 must start before t3.
        task_graph = TaskGraph.create(
            tasks=[("t1", 1.0), ("t2", 1.0), ("t3", 1.0)],
            dependencies=[("t1", "t2", 0.0), ("t2", "t3", 0.0)],
        )
        schedule = FIFOEnvironment(network, task_graph).run()
        t2_start = None
        t3_start = None
        for tasks in schedule.mapping.values():
            for t in tasks:
                if t.name == "t2":
                    t2_start = t.start
                if t.name == "t3":
                    t3_start = t.start
        assert t2_start is not None and t3_start is not None
        assert t2_start < t3_start

    def test_diamond_concurrent_tasks_dispatched_in_fifo_order(
        self, two_node_network, diamond_task_graph
    ):
        """t2 and t3 both become ready when t1 finishes. On a single-step-dispatch
        schedule the one inserted first (earlier in heap, tiebroken by name) must be
        scheduled before the other."""
        schedule = FIFOEnvironment(two_node_network, diamond_task_graph).run()
        # t2 and t3 both become ready at t1.end. With a min-heap keyed on (time, name),
        # "t2" < "t3" alphabetically, so t2 is dispatched first.
        t2_start, t3_start = None, None
        for tasks in schedule.mapping.values():
            for t in tasks:
                if t.name == "t2":
                    t2_start = t.start
                elif t.name == "t3":
                    t3_start = t.start
        assert t2_start is not None and t3_start is not None
        assert t2_start <= t3_start


# ---------------------------------------------------------------------------
# Integration — no node overlap
# ---------------------------------------------------------------------------

class TestNoNodeOverlap:
    def test_chain_no_overlap(self, two_node_network, chain_task_graph):
        schedule = FIFOEnvironment(two_node_network, chain_task_graph).run()
        for node_name, tasks in schedule.mapping.items():
            for i in range(len(tasks) - 1):
                assert tasks[i].end <= tasks[i + 1].start + 1e-9, (
                    f"Overlap on {node_name}: {tasks[i]} and {tasks[i+1]}"
                )

    def test_diamond_no_overlap(self, two_node_network, diamond_task_graph):
        schedule = FIFOEnvironment(two_node_network, diamond_task_graph).run()
        for node_name, tasks in schedule.mapping.items():
            for i in range(len(tasks) - 1):
                assert tasks[i].end <= tasks[i + 1].start + 1e-9, (
                    f"Overlap on {node_name}: {tasks[i]} and {tasks[i+1]}"
                )
