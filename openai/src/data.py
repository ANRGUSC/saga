from saga.schedulers import (BILScheduler, CpopScheduler, DuplexScheduler,
                             ETFScheduler, FastestNodeScheduler, FCPScheduler,
                             FLBScheduler, GDLScheduler, HeftScheduler,
                             MaxMinScheduler, MCTScheduler, METScheduler,
                             MinMinScheduler, OLBScheduler, WBAScheduler, SDBATSScheduler)

# scheduler map
SCHEDULER_MAP = {
    1: BILScheduler(),
    2: CpopScheduler(),
    3: DuplexScheduler(),
    4: ETFScheduler(),
    5: FastestNodeScheduler(),
    6: FCPScheduler(),
    7: FLBScheduler(),
    8: GDLScheduler(),
    9: HeftScheduler(),
    10: MaxMinScheduler(),
    11: MCTScheduler(),
    12: METScheduler(),
    13: MinMinScheduler(),
    14: OLBScheduler(),
    15: WBAScheduler(),
    16: SDBATSScheduler()
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
TASK_GRAPH_DESCRIPTION  = (
        "(Network G = (T, D), where T is the set of tasks and D contains "
        "the directed edges or dependencies between these tasks? An edge (t, t′) ∈ D implies that the output "
        "from task t is required input for task t′.)"
)

NETWORK_GRAPH_DESCRIPTION = (
        "(N = (V, E) denote the compute node network, "
        "where N is a complete undirected graph. V is the set of nodes and E is the set of edges. The compute speed "
        "of a node v ∈ V is s(v) ∈ R+ and the communication strength between nodes (v,v′) ∈ E is s(v,v′) ∈ R+). "
)

# algorithm 
SCHEDULER_DESCRIPTION_MAP = {
    1: "(BILScheduler: Balanced Insertion Load scheduler that distributes tasks by minimizing the impact of inserting a task into a partially filled schedule. It dynamically balances load while considering node availability and task weight.)",
    
    2: "(CpopScheduler: Critical Path on a Processor scheduler that assigns all tasks on the critical path to a single processor to minimize inter-processor communication delays. It combines upward and downward rank values for accurate task prioritization.)",
    
    3: "(DuplexScheduler: Uses a bidirectional scheduling strategy where tasks are first scheduled in topological order (forward), and then refined in reverse topological order (backward), combining global and local optimization for better parallelism and reduced makespan.)",
    
    4: "(ETFScheduler: Earliest Time First scheduler that selects the task which can start at the earliest possible time and schedules it on the processor that provides the earliest finish time, promoting processor utilization in early stages of execution.)",
    
    5: "(FastestNodeScheduler: Assigns all tasks to the processor with the highest computational speed, ignoring communication delays. It's extremely naive but establishes an upper-bound reference for purely compute-optimized strategies.)",
    
    6: "(FCPScheduler: Finish-Communication Penalty scheduler that assigns tasks by minimizing the sum of finish time and communication overhead. It balances computational latency with inter-node data transfer penalties.)",
    
    7: "(FLBScheduler: Flexible Load Balancer that adapts dynamically to current system states by evaluating both runtime availability and task execution estimates. It continuously updates priorities as tasks are placed.)",
    
    8: "(GDLScheduler: Greedy Deadline-based scheduler that selects the processor minimizing a cost function combining computation time and communication cost. It operates locally but aims to achieve global efficiency in execution time.)",
    
    9: "(HeftScheduler: Heterogeneous Earliest Finish Time (HEFT) algorithm that uses an upward rank heuristic to prioritize tasks and assigns each task to the processor minimizing earliest finish time. Widely used and well-established for DAG-based scheduling on heterogeneous systems.)",
    
    10: "(MaxMinScheduler: For each unscheduled task, selects the processor that provides the minimum completion time. Then among all such minimums, chooses the task with the **maximum** of those values to assign. This tends to help long tasks finish earlier and promotes balance.)",
    
    11: "(MCTScheduler: Minimum Completion Time scheduler assigns each task to the processor where it will finish the earliest, evaluated independently for each task without global task ranking. Simple and fast, but can lead to imbalance.)",
    
    12: "(METScheduler: Minimum Execution Time scheduler selects the processor that provides the **minimum raw execution time**, regardless of communication or current load. It works well in systems with uniform communication but high variance in computation speed.)",
    
    13: "(MinMinScheduler: For each unscheduled task, selects the processor with the minimum completion time. Then chooses the task with the **minimum** of those values to assign. Prioritizes shorter tasks and minimizes initial makespan quickly.)",
    
    14: "(OLBScheduler: Opportunistic Load Balancing assigns each task to the first available processor regardless of expected execution time or load. It ensures high utilization but performs poorly in minimizing makespan.)",
    
    15: "(WBAScheduler: Weighted Balanced Assignment scheduler assigns tasks based on a scoring system that weighs both processor speed and task complexity, aiming to distribute the load proportionally across available resources.)",
    
    16: "(SDBATSScheduler: Stochastic Dynamic Budget-Aware Task Scheduler uses probabilistic models and cost estimation to make adaptive scheduling decisions, balancing runtime uncertainty, cost constraints, and execution time.)"
}