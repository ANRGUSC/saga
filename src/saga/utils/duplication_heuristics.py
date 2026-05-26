from saga import TaskGraph
import networkx as nx


def is_super_node(task_name: str) -> bool:
    # HELPER: ignore fake nodes (super source/sinks) during scoring
    return task_name.startswith("__super_")

def task_can_be_scored(task_name: str, task_graph: TaskGraph) -> bool: 
    # HELPER: task cannot be scored if it has no children
    if is_super_node(task_name): return False
    real_children = [
        edge.target for edge in task_graph.out_edges(task_name)
        if not is_super_node(edge.target)
    ]
    return len(real_children) > 0

def communication_score(task_name: str, task_graph: TaskGraph) -> float:
    # a task with a larger incoming_sum would receive a higher score  
    if not task_can_be_scored(task_name, task_graph): return 0.0

    outgoing_sum = sum(
        edge.size for edge in task_graph.out_edges(task_name)
        if not is_super_node(edge.target)
    )

    incoming_sum = sum(
        edge.size for edge in task_graph.in_edges(task_name)
        if not is_super_node(edge.source)
    )

    if outgoing_sum + incoming_sum == 0: return 0.0

    return max(0.0, (outgoing_sum - incoming_sum) / (outgoing_sum + incoming_sum)) 

def impact_score(task_name: str, task_graph: TaskGraph) -> float:
    # impact_score = descendant_computation / total_graph_computation
    if is_super_node(task_name) or not task_can_be_scored(task_name, task_graph):
        return 0.0

    descendants = {
        descendant for descendant in nx.descendants(task_graph.graph, task_name)
        if not is_super_node(descendant)
    }
    
    descendant_compute = sum(
        task_graph.get_task(descendant).cost for descendant in descendants
    )

    total_graph_compute = sum(
        task.cost for task in task_graph.tasks
        if not is_super_node(task.name)
    )
    if total_graph_compute == 0:
        return 0.0  
    
    return descendant_compute / total_graph_compute

def branching_score(task_name: str, task_graph: TaskGraph) -> float:
    # branching_score = number_of_children / max_number_of_children
    if is_super_node(task_name) or not task_can_be_scored(task_name, task_graph):
        return 0.0

    task_out_degree = sum(
        1 for edge in task_graph.out_edges(task_name)
        if not is_super_node(edge.target)
    )

    max_out_degree = max(
        sum(
            1 for edge in task_graph.out_edges(task.name)
            if not is_super_node(edge.target)
        )
        for task in task_graph.tasks
        if not is_super_node(task.name)
    )

    if max_out_degree == 0:
        return 0.0

    return task_out_degree / max_out_degree

def join_val_score(task_name: str, task_graph: TaskGraph) -> float:
    # more than 1 parent meet back up at 1 task
    if not task_can_be_scored(task_name, task_graph): return 0.0

    scores = []
    for child_edge in task_graph.out_edges(task_name):
        child_name = child_edge.target
        if task_graph.in_degree(child_name) < 2: 
            continue
        
        incoming_edges = task_graph.in_edges(child_name)
        max_incoming = max(edge.size for edge in incoming_edges)
        if max_incoming == 0: 
            continue
        scores.append(child_edge.size / max_incoming)
    
    if not scores: 
        return 0.0
    
    return max(scores)

def task_score(task_name: str, task_graph: TaskGraph) -> float: 
    return (
        communication_score(task_name, task_graph) 
        + impact_score(task_name, task_graph) 
        + branching_score(task_name, task_graph) 
        + join_val_score(task_name, task_graph)
    ) / 4