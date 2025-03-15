import pytest
import traceback
import networkx as nx

from saga.scheduler import Scheduler
from saga.schedulers import (
    BruteForceScheduler, CpopScheduler, DuplexScheduler, ETFScheduler,
    FastestNodeScheduler, FCPScheduler, HeftScheduler, MaxMinScheduler,
    METScheduler, MinMinScheduler, SMTScheduler, WBAScheduler, HybridScheduler,
    BILScheduler, FLBScheduler, GDLScheduler, SDBATSScheduler
)
from saga.schedulers.stochastic.improved_sheft import ImprovedSheftScheduler
from saga.schedulers.stochastic.sheft import SheftScheduler
from saga.schedulers.stochastic.stoch_heft import StochHeftScheduler
from saga.utils.random_graphs import (
    add_random_weights,
    add_rv_weights,
    get_branching_dag,
    get_chain_dag,
    get_diamond_dag,
    get_fork_dag,
    get_network,
)
from saga.utils.tools import validate_simple_schedule

# set seeds for reproducibility
import random
import numpy as np

random.seed(0)
np.random.seed(0)

def run_test(scheduler: Scheduler,
             network: nx.Graph,
             task_graph: nx.DiGraph) -> bool:
    """Runs the test and validates the schedule."""
    try:
        schedule = scheduler.schedule(network, task_graph)
        validate_simple_schedule(network, task_graph, schedule)
        return True
    except Exception as exp:
        print(f"Error: {exp}\nStacktrace: {traceback.format_exc()}")
        return False

def run_stochastic_test(scheduler: Scheduler,
                        network: nx.Graph,
                        task_graph: nx.DiGraph) -> bool:
    """Runs the test and validates the schedule."""
    try:
        schedule = scheduler.schedule(network, task_graph)
        # TODO: validate stochastic schedule
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
    SDBATSScheduler(),
]

# Parametrize the task graphs for common schedulers
common_task_graphs = {
    "diamond": add_random_weights(get_diamond_dag()),
    "chain": add_random_weights(get_chain_dag()),
    "fork": add_random_weights(get_fork_dag()),
    "branching": add_random_weights(get_branching_dag(levels=5, branching_factor=3)),
}

# Parametrize the task graphs for stochastic schedulers
stochastic_task_graphs = {
    "diamond": add_rv_weights(get_diamond_dag()),
    "chain": add_rv_weights(get_chain_dag()),
    "fork": add_rv_weights(get_fork_dag()),
    "branching": add_rv_weights(get_branching_dag()),
}

# Stochastic schedulers
stochastic_schedulers = [
    SheftScheduler(),
    StochHeftScheduler(),
    ImprovedSheftScheduler(),
]

network = add_random_weights(get_network())
rv_network = add_rv_weights(get_network())


@pytest.mark.parametrize("scheduler", schedulers)
@pytest.mark.parametrize("task_graph_name, task_graph", common_task_graphs.items())
def test_common_schedulers(scheduler, task_graph_name, task_graph):
    """Test common schedulers on predefined task graphs."""
    assert run_test(scheduler, network.copy(), task_graph.copy()), f"Test failed for {scheduler.__class__.__name__} on {task_graph_name}"


@pytest.mark.parametrize("scheduler", stochastic_schedulers)
@pytest.mark.parametrize("task_graph_name, task_graph", stochastic_task_graphs.items())
def test_stochastic_schedulers(scheduler: Scheduler, task_graph_name: str, task_graph: nx.DiGraph):
    """Test stochastic schedulers on predefined task graphs."""
    assert run_stochastic_test(scheduler, rv_network.copy(), task_graph.copy()), f"Test failed for {scheduler.__class__.__name__} on {task_graph_name}"
