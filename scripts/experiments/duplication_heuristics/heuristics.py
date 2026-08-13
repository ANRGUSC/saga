"""Candidate scoring heuristics for choosing which tasks to duplicate.

Each score maps a task to a number in [0, 1]; higher means "more likely to be a
good duplication target". These are experimental:
they are evaluated by ``run.py``.
"""

from saga import TaskGraph, Network, TaskGraphEdge, Schedule, ScheduledTask

# HELPERS
def is_super_node(task_name: str) -> bool:
    """Whether a task is a synthetic super source/sink added by TaskGraph.create."""
    return task_name.upper() in {"SRC", "DST"} or task_name.startswith("__super_")

def task_can_be_scored(task_name: str, task_graph: TaskGraph) -> bool: 
    """
    Checks to see if a task can be scored. A task can't be scored if they are not real or have no children.
    """
    if is_super_node(task_name): return False
    real_children = [
        edge.target
        for edge in task_graph.out_edges(task_name)
        if not is_super_node(edge.target)
    ]
    return len(real_children) > 0

def normalize(value: float, max_value: float) -> float:
    """Normalizes scores to be between 0.0 to 1.0."""
    if max_value == 0: return 0.0
    return max(0.0, min(1.0, value / max_value))

def comm_time(edge: TaskGraphEdge, parent_task: ScheduledTask, child_task: ScheduledTask, network: Network) -> float:
    """Returns the comm delay to transfer data from a parent task to a child task."""
    if parent_task.node == child_task.node: return 0.0
    link_speed = network.get_edge(parent_task.node, child_task.node).speed
    return edge.size / link_speed

def get_task_instances(task_name: str, schedule: Schedule) -> list[ScheduledTask]:
    """Returns all scheduled instances of a task, including duplicates."""
    instances = schedule._task_map.get(task_name)
    if not instances: raise ValueError(f"No scheduled instances for task {task_name}")
    return list(instances)

def get_data_arrival_time(parent_copy: ScheduledTask, child_copy: ScheduledTask, edge: TaskGraphEdge, schedule: Schedule) -> float:
    """Returns the time when a parent's output becomes available to a specific child's processor."""
    comm_delay = comm_time(edge, parent_copy, child_copy, schedule.network)
    return parent_copy.end + comm_delay

def get_earliest_parent_copy(parent_name: str, child_copy: ScheduledTask, edge: TaskGraphEdge, schedule: Schedule) -> ScheduledTask:
    """Returns the parent task that provides data to the child earliest."""
    parent_copies = get_task_instances(parent_name, schedule)
    return min(parent_copies, key=lambda parent_copy: (
        get_data_arrival_time(parent_copy, child_copy, edge, schedule),
        parent_copy.end,
        parent_copy.start,
        parent_copy.node
    ))

def get_earliest_parent_child_pairs(edge: TaskGraphEdge, schedule: Schedule) -> list[tuple[ScheduledTask, ScheduledTask]]:
    """Returns the scheduled parent-child instance pairs that effectively communicate across a dependency edge."""
    parent_name = edge.source
    child_name = edge.target
    child_copies = get_task_instances(child_name, schedule)
    pairs: list[tuple[ScheduledTask, ScheduledTask]] = []

    for child_copy in child_copies:
        earliest_parent_copy = get_earliest_parent_copy(parent_name, child_copy, edge, schedule)
        pairs.append((earliest_parent_copy, child_copy))
    return pairs

def get_effective_inputs_for_child(child_copy: ScheduledTask, schedule: Schedule) -> list[tuple[TaskGraphEdge, ScheduledTask, float]]:
    """Returns the parent inputs used by a scheduled child copy, including each dependency and its data-arrival time."""
    task_graph = schedule.task_graph
    effective_inputs: list[tuple[TaskGraphEdge, ScheduledTask, float]] = []
    
    for incoming_edge in task_graph.in_edges(child_copy.name):
        parent_name = incoming_edge.source
        if is_super_node(parent_name):
            continue

        earliest_parent_copy = get_earliest_parent_copy(parent_name, child_copy, incoming_edge, schedule)
        arrival_time = get_data_arrival_time(earliest_parent_copy, child_copy, incoming_edge, schedule)
        effective_inputs.append((incoming_edge, earliest_parent_copy, arrival_time))
    return effective_inputs

def get_effective_children_for_parent_copy(parent_copy: ScheduledTask, schedule: Schedule) -> list[tuple[TaskGraphEdge, ScheduledTask, float]]:
    """
    Returns the child copies that receive data earliest from a specific parent copy, 
    including their communication arrival times.
    """
    task_graph = schedule.task_graph
    effective_children: list[tuple[TaskGraphEdge, ScheduledTask, float]] = []

    for outgoing_edge in task_graph.out_edges(parent_copy.name):
        child_name = outgoing_edge.target
        if is_super_node(child_name):
            continue
        child_copies = get_task_instances(child_name, schedule)
        for child_copy in child_copies:
            earliest_parent_copy = get_earliest_parent_copy(parent_copy.name, child_copy, outgoing_edge, schedule)
            if earliest_parent_copy is not parent_copy:
                continue

            arrival_time = get_data_arrival_time(parent_copy, child_copy, outgoing_edge, schedule)
            effective_children.append((outgoing_edge, child_copy, arrival_time))
    return effective_children

def get_next_task_on_proc(task_copy: ScheduledTask, schedule: Schedule) -> ScheduledTask | None:
    """
    Returns the next scheduled task on the same processor. Returns None if no later task exists on that processor.

    Used to determine whether placing a duplicate on that processor could interfere with later scheduled work.
    """
    proc_tasks = sorted(schedule.mapping.get(task_copy.node, []), key=lambda scheduled: (
        scheduled.start, scheduled.end, scheduled.name
    ))

    for i, scheduled_task in enumerate(proc_tasks):
        if scheduled_task is task_copy:
            return proc_tasks[i + 1] if i + 1 < len(proc_tasks) else None

    for i, scheduled_task in enumerate(proc_tasks):
        if (
            scheduled_task.name == task_copy.name and 
            scheduled_task.node == task_copy.node and 
            scheduled_task.start == task_copy.start and 
            scheduled_task.end == task_copy.end
        ):
            return proc_tasks[i + 1] if i + 1 < len(proc_tasks) else None
    
    return None

def compute_task_slack(schedule: Schedule) -> dict[str, float]:
    """
    Compute the scheduling slack of each task. 

    Slack is the amount a task can be delayed without increasing the workflow makespan.
    Lower slack indicates a more critical task. 
    """
    # a task is constrained by: child dependency timing, next task on same processor, and the schedule makespan
    task_graph = schedule.task_graph
    network = schedule.network
    makespan = schedule.makespan
    all_task_instances = [
        scheduled_task for proc_tasks in schedule.mapping.values() 
        for scheduled_task in proc_tasks 
        if not is_super_node(scheduled_task.name)
    ]

    reverse_task_intances = sorted(all_task_instances, key=lambda task: (task.end, task.start), reverse=True)
    # latest_finish_time
    LFT_by_instance: dict[int, float] = {}
    for task_copy in reverse_task_intances:
        constraints: list[float] = [makespan]
        # dependency constraints from child copies given by this parent copy
        effective_children_for_parent_copy = get_effective_children_for_parent_copy(task_copy, schedule)
        for edge, child_copy, _ in effective_children_for_parent_copy:
            child_latest_finish = LFT_by_instance.get(id(child_copy), makespan)
            child_proc = network.get_node(child_copy.node)
            child_task = task_graph.get_task(child_copy.name)
            child_runtime = child_task.cost / child_proc.speed
            comm_delay = comm_time(edge, task_copy, child_copy, network)
            constraints.append(child_latest_finish - child_runtime - comm_delay)
        
        # processor-order constraint
        next_task = get_next_task_on_proc(task_copy, schedule)
        if next_task is not None: 
            next_latest_finish = LFT_by_instance.get(id(next_task), makespan)
            next_runtime = next_task.end - next_task.start
            constraints.append(next_latest_finish - next_runtime)
        
        LFT_by_instance[id(task_copy)] = min(constraints)
    
    task_level_slacks: dict[str, float] = {}
    for task_copy in all_task_instances:
        latest_finish = LFT_by_instance[id(task_copy)]
        instance_slack = max(0.0, latest_finish - task_copy.end)
        if task_copy.name not in task_level_slacks:
            task_level_slacks[task_copy.name] = instance_slack
        else: 
            # the most constrained copy determines task urgency 
            task_level_slacks[task_copy.name] = min(task_level_slacks[task_copy.name], instance_slack)
    
    return task_level_slacks
    

# HEURISTICS
def communication_ratio_score(task_name: str, task_graph: TaskGraph, network: Network, schedule: Schedule) -> float:
    """
    Scores a task based on its incoming and outgoing communication costs.
    
    Higher scores are given to tasks who are dominant senders (more outgoing than incoming).
    """
    if not task_can_be_scored(task_name, task_graph): 
        return 0.0
    
    instance_scores: list[float] = []
    task_instances = get_task_instances(task_name, schedule)
    for task_copy in task_instances:

        outgoing_comm_time = 0.0
        effective_children_for_parent_copy = get_effective_children_for_parent_copy(task_copy, schedule)
        for edge, child_copy, _ in effective_children_for_parent_copy:
            outgoing_comm_time += comm_time(edge, task_copy, child_copy, network)
        
        incoming_comm_time = 0.0
        effective_inputs_for_child = get_effective_inputs_for_child(task_copy, schedule)
        for edge, parent_copy, _ in effective_inputs_for_child:
            incoming_comm_time += comm_time(edge, parent_copy, task_copy, network)

        total_comm = outgoing_comm_time + incoming_comm_time
        if total_comm <= 0.0:
            instance_scores.append(0.0)
            continue
        
        # direction_ratio: is this task more of a receiver or sender?
        direction_ratio = max(0.0, (outgoing_comm_time - incoming_comm_time) / total_comm)
        magnitude = outgoing_comm_time / (outgoing_comm_time + 1.0)
        # if both conditions are true, the score is high (dominant sender AND its outgoing transfers are large)
        instance_scores.append(direction_ratio * magnitude)
    return (sum(instance_scores) / len(instance_scores) if instance_scores else 0.0)

def cross_proc_branching_score(task_name: str, task_graph: TaskGraph, schedule: Schedule) -> float:
    """
    Score how much a task branches to children scheduled on different processors.

    Higher scores are given to tasks with more children scheduled on different processors, 
    relative to the max in the workflow.
    """
    if not task_can_be_scored(task_name, task_graph):
        return 0.0
    total_relationships, cross_proc_relationships = 0, 0
    for edge in task_graph.out_edges(task_name):
        if is_super_node(edge.target):
            continue
        
        parent_child_pairs = get_earliest_parent_child_pairs(edge, schedule)
        for parent_copy, child_copy in parent_child_pairs:
            total_relationships += 1
            if parent_copy.node != child_copy.node:
                cross_proc_relationships += 1
    
    if total_relationships == 0: 
        return 0.0
    
    return cross_proc_relationships / total_relationships 

def join_bottleneck_score(task_name: str, task_graph: TaskGraph, network: Network, schedule: Schedule) -> float:
    """
    Scores whether a task is a bottleneck parent for a downstream join task.

    Higher scores are given to a parents whose data arrives later than the other inputs, 
    causing the join task to wait.
    """
    # more than 1 parent meet back up at 1 task (join node/task)
    # a high score (slow finish time) means this task is the biggest reason why the join node has to wait (bottleneck parent)
    if not task_can_be_scored(task_name, task_graph): 
        return 0.0

    scores: list[float] = []
    for outgoing_edge in task_graph.out_edges(task_name):
        child_name = outgoing_edge.target
        if is_super_node(child_name) or task_graph.in_degree(child_name) < 2:
            continue
        
        task_instances = get_task_instances(child_name, schedule)
        for child_copy in task_instances:
            earliest_parent = get_earliest_parent_copy(task_name, child_copy, outgoing_edge, schedule)
            task_arrival = get_data_arrival_time(earliest_parent, child_copy, outgoing_edge, schedule)
            inputs_for_child = get_effective_inputs_for_child(child_copy, schedule)
            all_inputs_arrivals = [arrival_time for _, _, arrival_time in inputs_for_child]
            if not all_inputs_arrivals:
                continue

            latest_input_arrival = max(all_inputs_arrivals)
            # normalize relative to the child copy's actual input window
            earliest_input_arrival = min(all_inputs_arrivals)
            arrival_range = latest_input_arrival - earliest_input_arrival
            if arrival_range <= 0.0:
                # every input arrives together, so no single parent is the bottleneck
                scores.append(0.0)
                continue

            score = (task_arrival - earliest_input_arrival) / arrival_range
            scores.append(max(0.0, min(1.0, score)))
    return max(scores, default=0.0)

def slack_score(task_name: str, slacks: dict[str, float], makespan: float) -> float:
    """
    Convert a task's slack into a normalized urgency score. 

    Higher scores favor tasks with less scheduling flexibility.
    """
    # low slack (more urgent) = near critical path = higher score close to 1.0
    # high slack (less urgent) = not near critical path = lower score close to 0.0
    # slack = 0 -> score = 1.0 (exactly on critical path)

    if makespan <= 0.0: return 0.0
    slack = max(0.0, slacks.get(task_name, makespan))
    return max(0.0, min(1.0, 1.0 - (slack / makespan)))

def processor_dup_benefits(task_name: str, task_graph: TaskGraph, schedule: Schedule) -> list[tuple[str, float]]:
    """
    Estimates the benefit of duplicating a task onto processors containing its children.
    
    Returns target processors (child processors) and their estimated data-arrival benefit, 
    sorted from highest to lowest benefit.
    """
    network = schedule.network
    task = task_graph.get_task(task_name)
    task_instances = get_task_instances(task_name, schedule)
    all_processors = {task_copy.node for task_copy in task_instances}
    # group children by the processor containing their child
    children_by_processor: dict[str, list[tuple[TaskGraphEdge, ScheduledTask]]] = {}

    for edge in task_graph.out_edges(task_name):
        child_name = edge.target
        if is_super_node(child_name): 
            continue
        child_task_instances = get_task_instances(child_name, schedule)
        for child_copy in child_task_instances:
            children_by_processor.setdefault(child_copy.node, []).append((edge, child_copy))
        
    
    benefit_per_proc: dict[str, float] = {}
    for target_proc_name, child_entries in children_by_processor.items():
        # do not recommend another duplicate on a processor that already contains this task
        if target_proc_name in all_processors:
            continue
        
        target_proc = network.get_node(target_proc_name)
        dup_start = schedule.get_earliest_start_time(task, target_proc, False)
        dup_runtime = task.cost / target_proc.speed
        dup_data_ready = dup_start + dup_runtime
        total_benefit = 0.0

        for edge, child_copy in child_entries:
            curr_parent_copy = get_earliest_parent_copy(task_name, child_copy, edge, schedule)
            curr_data_arrival = get_data_arrival_time(curr_parent_copy, child_copy, edge, schedule)
            # if positive, duplication helps. vice versa
            child_benefit = curr_data_arrival - dup_data_ready
            if child_benefit > 0.0:
                total_benefit += child_benefit
        
        if total_benefit > 0.0:
            benefit_per_proc[target_proc_name] = total_benefit
    # returns target processors and their estimated data-arrival benefit sorted by highest to lowest benefit 
    return sorted(benefit_per_proc.items(), key=lambda item: item[1], reverse=True) # return example: [("P2", 10.0), ..]

# COMPUTE SCHEDULE FEATURES + TASK SCORE
def compute_schedule_features(task_graph: TaskGraph, schedule: Schedule) -> dict:
    """
    Compute the schedule-based features used by the duplication heuristics, once per scheduling iteration.

    For each candidate task, compute the schedule-based values that are reused across multiple heuristics, 
    such as slack and processor duplication benefits. Also returns the max feature values used for normalization.
    """
    candidate_tasks = [
        task.name for task in task_graph.tasks
        if not is_super_node(task.name) and task_can_be_scored(task.name, task_graph)
    ]

    slacks = compute_task_slack(schedule)
    makespan = schedule.makespan
    raw = {}
    for task_name in candidate_tasks:
        target_processors = processor_dup_benefits(task_name, task_graph, schedule)

        raw[task_name] = {
            "slack": slack_score(task_name, slacks, makespan),
            "estimated_benefit": sum(benefit for _, benefit in target_processors),
            "num_target_processors": len(target_processors)
        }

    max_values = {
        "estimated_benefit": max((values["estimated_benefit"] for values in raw.values()), default=0.0),
        "num_target_processors": max((values["num_target_processors"] for values in raw.values()), default=0.0)
    }

    return {"raw": raw, "max": max_values}

def task_score(task_name: str, task_graph: TaskGraph, network: Network, schedule: Schedule, schedule_features: dict) -> float: 
    """
    Compute a task's overall duplication score from the individual heuristics.

    Higher scores indicate stronger candidates for duplication.
    """
    raw = schedule_features["raw"][task_name]
    max_vals = schedule_features["max"]
    urgency = raw["slack"]
    estimated_benefit = normalize(raw["estimated_benefit"], max_vals["estimated_benefit"])

    graph_aware_score = (
        communication_ratio_score(task_name, task_graph, network, schedule) 
        + cross_proc_branching_score(task_name, task_graph, schedule) 
        + join_bottleneck_score(task_name, task_graph, network, schedule)
    ) / 3

    schedule_aware_score = (urgency + estimated_benefit) / 2
    # this is the current weighted scoring. more trials will be done to find out ideal weight
    final_score = (0.35 * graph_aware_score) + (0.65 * schedule_aware_score)
    return max(0.0, min(1.0, final_score))