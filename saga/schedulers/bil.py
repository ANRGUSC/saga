import copy
from typing import Dict, Hashable, List, Tuple

import networkx as nx

from ..scheduler import Scheduler, Task


class BILScheduler(Scheduler): # pylint: disable=too-few-public-methods
    """Best Imaginary Level Scheduler

    Source: https://doi.org/10.1007/BFb0024750
    Modifications:
    - The original algorithm does not consider heterogenous communication strengths between
      network nodes. This affects the ipc_overhead term in the BIL definition. We compute the
      ipc_overhead between two tasks by scaling the cost by the average communication weight
      in the network.
        Original: BIL(task, node) = task_cost / node_speed + max(BIL(child, node) + ipc_overhead)
                                                                 for child in children(task))
    """
    def schedule(self, # pylint: disable=too-many-locals
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Returns the schedule for the given task graph on the given network using the BIL algorithm.

        Args:
            network: Network
            task_graph: Task graph

        Returns:
            A dictionary of the schedule
        """
        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        scheduled_tasks: Dict[Hashable, Task] = {}

        bils: Dict[Hashable, float] = {}
        for task in reversed(list(nx.topological_sort(task_graph))):
            for node in network.nodes:
                exec_time = task_graph.nodes[task]['weight'] / network.nodes[node]['weight']
                bils[(task, node)] = exec_time + max(
                    (
                        min(
                            bils[(child, node)],
                            *(
                                bils[(child, other_node)] + (task_graph.edges[task, child]['weight'] /
                                                             network.edges[other_node, node]['weight'])
                                for other_node in network.nodes
                            )
                        )
                        for child in task_graph.successors(task)
                    ),
                    default=0
                )

        ready_tasks = {task for task in task_graph.nodes if task_graph.in_degree(task) == 0}
        while len(scheduled_tasks) < len(task_graph.nodes):
            # Section 3.1: Node Selection
            bims: Dict[Hashable, List[Tuple[Hashable, float]]] = {
                task: sorted([
                    (node, (schedule[node][-1].end if schedule[node] else 0) + bils[(task, node)])
                    for node in network.nodes
                ], key=lambda x: x[1], reverse=True)
                for task in ready_tasks if task not in scheduled_tasks
            }

            j = 0
            selected_tasks = copy.copy(ready_tasks)
            while j < len(network.nodes):
                max_bim = -1
                selected_tasks = set()
                for task, _bims in bims.items():
                    _, bim = _bims[j]
                    if bim == max_bim:
                        selected_tasks.add(task)
                    elif bim > max_bim:
                        max_bim = bim
                        selected_tasks = {task}
                if len(selected_tasks) == 1:
                    break
                j += 1

            selected_task = selected_tasks.pop()

            # Section 3.2: Processor Selection
            # compute revised bims for selected task and all nodes
            revised_bims: Dict[Hashable, float] = {
                node: (
                    (schedule[node][-1].end if schedule[node] else 0) +
                    bils[(selected_task, node)] +
                    (
                        task_graph.nodes[selected_task]['weight'] / network.nodes[node]['weight'] *
                        max(len(ready_tasks)/len(task_graph.nodes)-1,0)
                    )
                )
                for node in network.nodes
            }

            # select node with lowest revised bim
            selected_node = min(revised_bims, key=revised_bims.get)
            # If more than one processor have the same revised BIM value, we select the
            # processor that makes the sum of the revised BIM values of other nodes on the
            # processor maximum.
            if len([node for node in network.nodes if revised_bims[node] == revised_bims[selected_node]]) > 1:
                selected_node = max(network.nodes, key=lambda node: sum(
                    revised_bims[other_node] for other_node in network.nodes if other_node != node
                ))

            # Schedule
            start_time = max(
                schedule[selected_node][-1].end if schedule[selected_node] else 0,
                max((
                    scheduled_tasks[child].end + (
                        task_graph.edges[child, selected_task]['weight'] /
                        network.edges[selected_node, scheduled_tasks[child].node]['weight']
                    )
                    for child in task_graph.predecessors(selected_task)        
                ), default=0)
            )
            end_time = start_time + task_graph.nodes[selected_task]['weight'] / network.nodes[selected_node]['weight']
            new_task = Task(
                node=selected_node,
                name=selected_task,
                start=start_time,
                end=end_time
            )
            schedule[selected_node].append(new_task)
            scheduled_tasks[selected_task] = new_task

            # Update ready tasks
            ready_tasks.update({
                child for child in task_graph.successors(selected_task)
                if all(parent in scheduled_tasks for parent in task_graph.predecessors(child))
            })

        return schedule
