from typing import Dict, Hashable, List, Tuple
import networkx as nx

from ....scheduler import Scheduler, Task

class ResidualOLBScheduler(Scheduler): # pylint: disable=too-few-public-methods
    """Opportunistic Load Balancing scheduler

    Source: https://doi.org/10.1006/jpdc.2000.1714
    Summary: "(OLB) assigns each task, in arbitrary order, to the next machine that is expected
        to be available, regardless of the task's expected execution time on that machine"
        (from source).
    """
    def schedule(self, network: nx.Graph, task_graphs: List[Tuple[nx.DiGraph, float]],) -> Dict[Hashable, List[Task]]:
        """Schedule tasks on nodes using the OLB algorithm.

        Args:
            network (nx.Graph): Network topology.
            task_graph (nx.DiGraph): Task graph.

        Returns:
            Dict[Hashable, List[Task]]: Schedule of the tasks on the network.
        """
        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        scheduled_tasks: Dict[Hashable, Task] = {}


        for task_graph_tupple in task_graphs:
            task_graph = task_graph_tupple[0]
            task_graph_arrival_time = task_graph_tupple[1]


            for task in nx.topological_sort(task_graph):
                next_available_node = min(
                    network.nodes,
                    key=lambda node: schedule[node][-1].end if schedule[node] else 0
                )
                times = [
                    # time node is available
                    schedule[next_available_node][-1].end if schedule[next_available_node] else 0,
                    *[
                        scheduled_tasks[predecessor].end + (
                            task_graph.edges[predecessor, task]['weight'] /
                            network.edges[scheduled_tasks[predecessor].node, next_available_node]['weight']
                        )
                        for predecessor in task_graph.predecessors(task)
                    ], task_graph_arrival_time
                ]
                start_time = max(times)
                exec_time = task_graph.nodes[task]['weight'] / network.nodes[next_available_node]['weight']
                new_task = Task(
                    name=task,
                    node=next_available_node,
                    start=start_time,
                    end=start_time + exec_time
                )

                schedule[next_available_node].append(new_task)
                scheduled_tasks[task] = new_task

        return schedule
