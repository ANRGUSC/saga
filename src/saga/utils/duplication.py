import numpy as np


from saga import Network, TaskGraph


def should_duplicate(task_name: str, task_graph: TaskGraph, network: Network) -> bool:
    """Determine if a task should be duplicated based on computation vs communication costs.

    Args:
        task_name (str): The name of the task.
        task_graph (TaskGraph): The task graph.
        network (Network): The network graph.

    Returns:
        bool: True if the average communication time exceeds the average computation time.
    """
    if not task_graph.out_edges(task_name):
        return False

    task = task_graph.get_task(task_name)
    average_computation_time = np.mean(
        [task.cost / node.speed for node in network.nodes]
    )

    # Average over usable inter-node links only: self-loops (local memory) and links
    # with non-positive speed (Network.create fills absent node pairs with speed 0)
    # are excluded so they don't distort or zero out the mean.
    link_speeds = [
        edge.speed
        for edge in network.edges
        if edge.source != edge.target and edge.speed > 0
    ]
    if not link_speeds:
        return False
    average_link_speed = np.mean(link_speeds)

    average_communication_time = np.mean(
        [
            dependency.size / average_link_speed
            for dependency in task_graph.out_edges(task_name)
        ]
    )
    return float(average_communication_time) > float(average_computation_time)
