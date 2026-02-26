import numpy as np


from saga import Network, TaskGraph


def should_duplicate(task_name: str, task_graph: TaskGraph, network: Network) -> bool:
    """Determine if a task should be duplicated based on computation vs communication costs.

    Args:
        task_name (Hashable): The name of the task.
        task_graph (TaskGraph): The task graph.
        network (Network): The network graph.

    Returns:
        bool: True if the average communication time exceeds the average computation time.
    """
    task = task_graph.get_task(task_name)
    average_computation_time = np.mean(
        [task.cost / node.speed for node in network.nodes]
    )

    # Calculate average outgoing communication cost
    # For each child task, estimate the communication time based on average network speed
    # if len(task_graph.out_edges(task_name))  < 2:
    #     return False
    if not task_graph.out_edges(task_name):
        return False

    # Use only inter-node links (exclude self-loops which represent local memory)
    inter_node_links = [edge for edge in network.edges if edge.source != edge.target]
    if not inter_node_links:
        return False

    average_link_speed = np.mean([edge.speed for edge in inter_node_links])

    comm_costs = []
    for dependency in task_graph.out_edges(task_name):
        # Communication time for this edge using average network speed
        comm_time = dependency.size / average_link_speed
        comm_costs.append(comm_time)

    average_communication_time = np.mean(comm_costs)
    return float(average_communication_time) > float(average_computation_time)
