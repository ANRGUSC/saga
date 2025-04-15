from saga.schedulers import (BILScheduler, CpopScheduler, DuplexScheduler,
                             ETFScheduler, FastestNodeScheduler, FCPScheduler,
                             FLBScheduler, GDLScheduler, HeftScheduler,
                             MaxMinScheduler, MCTScheduler, METScheduler,
                             MinMinScheduler, OLBScheduler, WBAScheduler, SDBATSScheduler)

# scheduler map
SCHEDULER_MAP = {
    1: BILScheduler,
    2: CpopScheduler,
    3: DuplexScheduler,
    4: ETFScheduler,
    5: FastestNodeScheduler,
    6: FCPScheduler,
    7: FLBScheduler,
    8: GDLScheduler,
    9: HeftScheduler,
    10: MaxMinScheduler,
    11: MCTScheduler,
    12: METScheduler,
    13: MinMinScheduler,
    14: OLBScheduler,
    15: WBAScheduler,
    16: SDBATSScheduler
}

# scheduler name map
SCHEDULER_NAME_MAP = {
    1: "BILScheduler",
    2: "CpopScheduler",
    3: "DuplexScheduler",
    4: "ETFScheduler",
    5: "FastestNodeScheduler",
    6: "FCPScheduler",
    7: "FLBScheduler",
    8: "GDLScheduler",
    9: "HeftScheduler",
    10: "MaxMinScheduler",
    11: "MCTScheduler",
    12: "METScheduler",
    13: "MinMinScheduler",
    14: "OLBScheduler",
    15: "WBAScheduler",
    16: "SDBATSScheduler"
}

# prompt descriptions

# input graph
NETWORK_GRAPH_DESCRIPTION = (
        "(Network G = (T, D), where T is the set of tasks and D contains "
        "the directed edges or dependencies between these tasks? An edge (t, t′) ∈ D implies that the output "
        "from task t is required input for task t′.)"
)

TASK_GRAPH_DESCRIPTION = (
        "(N = (V, E) denote the compute node network, "
        "where N is a complete undirected graph. V is the set of nodes and E is the set of edges. The compute speed "
        "of a node v ∈ V is s(v) ∈ R+ and the communication strength between nodes (v,v′) ∈ E is s(v,v′) ∈ R+). "
)

# algorithm 
SCHEDULER_DESCRIPTION_MAP = {
    1: "(BILScheduler: Balances incoming loads using task priorities and local node metrics.)",
    2: "(CpopScheduler: Chooses the critical path and assigns tasks to optimize overall path completion.)",
    3: "(DuplexScheduler: Combines forward and backward traversal to refine task order and improve parallelism.)",
    4: "(ETFScheduler: Uses the Earliest Time First strategy to minimize the earliest finish time for each task.)",
    5: "(FastestNodeScheduler: Assigns all tasks to the fastest available node, ignoring communication cost.)",
    6: "(FCPScheduler: Focuses on balancing communication and processing by looking at finish and communication penalties.)",
    7: "(FLBScheduler: A flexible load balancer that dynamically adapts to runtime metrics during task assignment.)",
    8: "(GDLScheduler: Greedy algorithm that locally selects nodes with the shortest execution and transfer time.)",
    9: "(HeftScheduler: HEFT uses upward ranking and processor selection to minimize makespan efficiently.)",
    10: "(MaxMinScheduler: Assigns the task with the maximum minimum completion time to its optimal node.)",
    11: "(MCTScheduler: Chooses the node with Minimum Completion Time for each task independently.)",
    12: "(METScheduler: Focuses on minimizing earliest task start times based on current resource availability.)",
    13: "(MinMinScheduler: Assigns tasks that can complete earliest first, favoring shorter tasks.)",
    14: "(OLBScheduler: Opportunistic Load Balancing assigns tasks randomly to free nodes, ignoring execution time.)",
    15: "(WBAScheduler: Weighted balancing approach using node speed and task complexity.)",
    16: "(SDBATSScheduler: A stochastic dynamic algorithm that adapts scheduling decisions based on budget and cost modeling.)"
}