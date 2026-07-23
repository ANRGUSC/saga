import copy
from typing import Dict, Hashable, List, Tuple

import networkx as nx

from ....scheduler import Scheduler, Task, DWScheduler


class ResidualBILScheduler(DWScheduler): # pylint: disable=too-few-public-methods
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
                 task_graphs: List[Tuple[nx.DiGraph, float]]
                 ) -> Dict[Hashable, List[Task]]:
        """Returns the schedule for the given task graph on the given network using the BIL algorithm.

        Args:
            network: Network
            task_graph: Task graph

        Returns:
            A dictionary of the schedule
        """
        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        

        # ===========================================
        for task_graph_tupple in task_graphs:
            task_graph = task_graph_tupple[0]
            task_graph_arrival_time = task_graph_tupple[1]

        # ===========================================
            scheduled_tasks: Dict[Hashable, Task] = {}
            bils: Dict[Hashable, float] = {}
            for task in reversed(list(nx.topological_sort(task_graph))):
                for node in network.nodes:
                    exec_time = task_graph.nodes[task]['weight'] / network.nodes[node]['weight']
                    bils[(task, node)] = exec_time + max(
                        (
                            # Here we are trying find find the minimum if we execute child on the same node or execute it on 
                            # another node and then communicate the result to the current node. We use minumum to find the best
                            # Then we take maximum of this computaion for all the children of the task to find the Best Imaginary Level
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
                        # if there are no children, we take task_graph_arrival_time as the default value
                        default=task_graph_arrival_time
                    )

            # we start with the tasks that have no dependencies (in_degree = 0)
            ready_tasks = {task for task in task_graph.nodes if task_graph.in_degree(task) == 0}
            while len(scheduled_tasks) < len(task_graph.nodes):
                # We keep scheduling tasks until all tasks are scheduled

                # Section 3.1: Node (Task) Selection
                # BIM is a dictionary of tasks and their BIM values for all nodes
                # the list consists of tuples of (node, bim) sorted in descending order of BIM values
                bims: Dict[Hashable, List[Tuple[Hashable, float]]] = {
                    # We try to find Best Imaginary Makespan BIM
                    task: sorted([
                        # shcedule[node][-1].end is the end time of the last task scheduled on the node
                        # if the node is not scheduled, we take task_graph_arrival_time as the end time
                        # we add the BIL value of the task on the node to the end time to get the BIM value
                        (node, (schedule[node][-1].end if schedule[node] else task_graph_arrival_time) + bils[(task, node)])
                        for node in network.nodes
                    # we just sort the BIM values in descending order by tuple[1] which is the BIM value  
                    ], key=lambda x: x[1], reverse=True)
                    # We only consider tasks that are ready and not scheduled
                    for task in ready_tasks if task not in scheduled_tasks
                }


                # Tie breaker
                # Here we are trying to find the task with the maximum BIM value.
                # We define the priority of a node as the k-th smallest BIM value or the largest finite BIM value if the k-th smallest BIM value is undefined. In case more than one node have the same priority, we adopt a tie breaking policy in a recursive form: we compare the (k - 1)-th BIMs of nodes that have the same k-th BIM
                # until we find a unique task with the maximum BIM value.
                # and if there are still multiple tasks, we just use pop to select one of them
                j = 0
                selected_tasks = copy.copy(ready_tasks)
                while j < len(network.nodes):
                    max_bim = -1
                    selected_tasks = set()
                    # iterate over all tasks and their BIM values
                    # add tasks with the maximum BIM value to the selected tasks
                    # if there is only one task with the maximum BIM value, we break the loop
                    for task, _bims in bims.items():
                        # _bims is a list of tuples of (node, bim) sorted in descending order of BIM values
                        # So, we are consider the BIM value of the task on the jth processor
                        _, bim = _bims[j]
                        if bim == max_bim:
                            # if the BIM value is same as the max BIM value, we add the task to the selected tasks
                            selected_tasks.add(task)
                        elif bim > max_bim:
                            # if the BIM value is greater than the max BIM value, we update the max BIM value and the selected tasks
                            max_bim = bim
                            selected_tasks = {task}
                    if len(selected_tasks) == 1:
                        # if we have only one task with the maximum BIM value, we break the loop
                        break
                    j += 1

                selected_task = selected_tasks.pop()

                # Section 3.2: Processor Selection
                # compute revised bims for selected task and all nodes
                revised_bims: Dict[Hashable, float] = {
                    node: (
                        # schedule[node][-1].end is the end time of the last task scheduled on the node
                        (schedule[node][-1].end if schedule[node] else task_graph_arrival_time) +
                        # we add the BIL value of the task on the node to the end time to get the BIM value
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
                    schedule[selected_node][-1].end if schedule[selected_node] else task_graph_arrival_time,
                    max((
                        scheduled_tasks[child].end + (
                            task_graph.edges[child, selected_task]['weight'] /
                            network.edges[selected_node, scheduled_tasks[child].node]['weight']
                        )
                        for child in task_graph.predecessors(selected_task)        
                    ), default= task_graph_arrival_time)
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
