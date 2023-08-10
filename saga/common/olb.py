from typing import Dict, Hashable, List
import networkx as nx

from ..base import Scheduler, Task

class OLBScheduler(Scheduler): # pylint: disable=too-few-public-methods
    """Opportunistic Load Balancing scheduler

    Source: https://doi.org/10.1006/jpdc.2000.1714
    """
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Schedule tasks on nodes using the OLB algorithm.

        Args:
            network (nx.Graph): Network topology.
            task_graph (nx.DiGraph): Task graph.

        Returns:
            Dict[Hashable, List[Task]]: Schedule of the tasks on the network.
        """
        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        scheduled_tasks: Dict[Hashable, Task] = {}
        for task in nx.topological_sort(task_graph):
            next_available_node = min(
                network.nodes,
                key=lambda node: schedule[node][-1].end if schedule[node] else 0
            )
            start_time = max(
                # time node is available
                schedule[next_available_node][-1].end if schedule[next_available_node] else 0,
                # time predecessor tasks are finished + communication time
                max(
                    schedule[predecessor][-1].end + (
                        task_graph.edges[predecessor, task]['weight'] /
                        network.edges[scheduled_tasks[predecessor].node, next_available_node]['weight']
                    )
                    for predecessor in task_graph.predecessors(task)
                ) if task_graph.predecessors(task) else 0
            )
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
