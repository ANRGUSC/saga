
import networkx as nx
from typing import Callable, Dict, List
from sched_scenes.common import Task, standardize_instance

def get_best_of_schedule(*get_schedules: Callable[[nx.Graph, nx.DiGraph], Dict[str, List[Task]]]) -> Callable[[nx.Graph, nx.DiGraph], Dict[str, List[Task]]]:
    """Returns a schedule function that returns the best schedule of the given schedule functions.

    Args:
        get_schedule (Callable[[nx.Graph, nx.DiGraph], Dict[str, List[Task]]]): The schedule functions.

    Returns:
        Callable[[nx.Graph, nx.DiGraph], Dict[str, List[Task]]]: A schedule function that returns the best schedule of the given schedule functions.
    """
    def get_best_of_schedule(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[str, List[Task]]:
        """Returns the best schedule of the given schedule functions.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[str, List[Task]]: The best schedule.
        """
        network, task_graph = standardize_instance(network, task_graph)
        best_schedule = None
        best_makespan = float("inf")
        for get_schedule in get_schedules:
            schedule = get_schedule(network, task_graph)
            makespan = max(task.end for tasks in schedule.values() for task in tasks)
            if makespan < best_makespan:
                best_schedule = schedule
                best_makespan = makespan
        return best_schedule
    return get_best_of_schedule