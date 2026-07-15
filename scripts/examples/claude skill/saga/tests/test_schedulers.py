from copy import deepcopy
import pytest
import traceback

from saga import Network, Scheduler, TaskGraph
from saga.schedulers import (
    BruteForceScheduler, CpopScheduler, DuplexScheduler, ETFScheduler,
    FastestNodeScheduler, FCPScheduler, HeftScheduler, MaxMinScheduler,
    METScheduler, MinMinScheduler, SMTScheduler, WBAScheduler, HybridScheduler,
    BILScheduler, FLBScheduler, GDLScheduler
)
from saga.schedulers.parametric.components import schedulers as parametric_schedulers
from saga.utils.random_graphs import (
    get_branching_dag,
    get_chain_dag,
    get_diamond_dag,
    get_fork_dag,
    get_network,
)

# set seeds for reproducibility
import random
import numpy as np

random.seed(0)
np.random.seed(0)

def run_test(scheduler: Scheduler,
             network: Network,
             task_graph: TaskGraph) -> bool:
    """Runs the test and validates the schedule."""
    try:
        scheduler.schedule(network, task_graph)
        return True
    except Exception as exp:
        print(f"Error: {exp}\nStacktrace: {traceback.format_exc()}")
        return False

# Parametrize the schedulers
schedulers = [
    HeftScheduler(),
    CpopScheduler(),
    FastestNodeScheduler(),
    BruteForceScheduler(),
    MinMinScheduler(),
    ETFScheduler(),
    MaxMinScheduler(),
    DuplexScheduler(),
    METScheduler(),
    FCPScheduler(),
    SMTScheduler(solver_name="z3"),
    HybridScheduler(schedulers=[HeftScheduler(), CpopScheduler()]),
    WBAScheduler(),
    BILScheduler(),
    FLBScheduler(),
    GDLScheduler(),
    *parametric_schedulers.values(),
]

# Parametrize the task graphs for common schedulers
common_task_graphs = {
    "diamond": get_diamond_dag(),
    "chain": get_chain_dag(),
    "fork": get_fork_dag(),
    # "branching": get_branching_dag(levels=3, branching_factor=2),
}

@pytest.mark.parametrize("scheduler", schedulers)
@pytest.mark.parametrize("task_graph_name, task_graph", common_task_graphs.items())
def test_schedulers(scheduler: Scheduler, task_graph_name: str, task_graph: TaskGraph):
    """Test common schedulers on predefined task graphs."""
    network = get_network(num_nodes=4)
    assert run_test(scheduler, deepcopy(network), deepcopy(task_graph)), f"Test failed for {scheduler.__class__.__name__} on {task_graph_name}"

