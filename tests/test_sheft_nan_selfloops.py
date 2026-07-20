"""Tests that degenerate RandomVariables produce NaN-free statistics.

StochasticNetwork.create fills missing self-loops with infinite speed, and SHEFT
determinizes with mean + std, so those statistics have to stay finite for
all-infinite samples.
"""

import math

from saga.schedulers.stochastic.sheft import SheftScheduler
from saga.stochastic import StochasticNetwork, StochasticTaskGraph
from saga.utils.random_variable import RandomVariable


def test_std_of_infinite_sample_is_not_nan():
    assert RandomVariable(samples=[math.inf]).std() == 0.0
    assert RandomVariable(samples=[math.inf, math.inf]).std() == 0.0
    assert RandomVariable(samples=[math.inf]).variance() == 0.0


def test_std_of_normal_samples_unchanged():
    assert RandomVariable(samples=[1.0, 2.0]).std() == 0.5
    # All-equal finite samples are genuinely zero-variance too.
    assert RandomVariable(samples=[5.0, 5.0, 5.0]).variance() == 0.0


def test_mixed_finite_and_infinite_samples_are_not_masked():
    # Only the all-equal degenerate case is treated as zero variance; a real
    # mix of finite and infinite samples must not be silently reported as 0.
    assert RandomVariable(samples=[1.0, math.inf]).variance() != 0.0


def test_sheft_schedules_with_auto_added_infinite_self_loops():
    """SHEFT completes on an instance whose self-loops default to inf speed."""
    rv = lambda: RandomVariable(samples=[1.0, 2.0])
    network = StochasticNetwork.create(
        nodes=[("A", rv()), ("B", rv())],
        edges=[("A", "B", rv())],  # no explicit self-loops -> create() adds inf ones
    )
    task_graph = StochasticTaskGraph.create(
        tasks=[("t1", rv()), ("t2", rv()), ("t3", rv())],
        dependencies=[("t1", "t2", rv()), ("t2", "t3", rv())],
    )

    # EstimateStochasticScheduler returns (schedule, det_network, det_task_graph).
    schedule, _, _ = SheftScheduler().schedule(network, task_graph)

    placed = {t.name for tasks in schedule.mapping.values() for t in tasks}
    assert placed == {"t1", "t2", "t3"}
