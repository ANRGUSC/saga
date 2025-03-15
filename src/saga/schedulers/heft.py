import logging
import pathlib
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np

from ..scheduler import Scheduler, Task
from ..utils.tools import get_insert_loc
from .cpop import upward_rank

thisdir = pathlib.Path(__file__).resolve().parent

# find priority order
def heft_rank_sort(network: nx.Graph, task_graph: nx.DiGraph) -> List[Hashable]:
    """Sort tasks based on their rank (as defined in the HEFT paper).

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        List[Hashable]: The sorted list of tasks.
    """
    #  rank_u(v) = w(v) + \max_{t \in successors(v)} \left( c(v, t) + rank_u(t) \right)

    rank = upward_rank(network, task_graph)

    # guarantee the tasks would be scheduled in order
    topological_sort = {node: i for i, node in enumerate(reversed(list(nx.topological_sort(task_graph))))}

    # break ties: why does this matter? - for sorting
    rank = {node: (rank[node] + topological_sort[node]) for node in rank}

    # largest rank in the front (reverse)
    return sorted(list(rank.keys()), key=rank.get, reverse=True)


class HeftScheduler(Scheduler):
    """Schedules tasks using the HEFT algorithm.

    Source: https://dx.doi.org/10.1109/71.993206
    """

    @staticmethod
    def get_runtimes(
        network: nx.Graph, task_graph: nx.DiGraph
    ) -> Tuple[
        Dict[Hashable, Dict[Hashable, float]],
        Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]],
    ]:
        """Get the expected runtimes of all tasks on all nodes.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Tuple[Dict[Hashable, Dict[Hashable, float]],
                  Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]]:
                A tuple of dictionaries mapping nodes to a dictionary of tasks and their runtimes
                and edges to a dictionary of tasks and their communication times. The first dictionary
                maps nodes to a dictionary of tasks and their runtimes. The second dictionary maps edges
                to a dictionary of task dependencies and their communication times.
        """

        # each node's runtime with all tasks
        runtimes = {}
        for node in network.nodes:
            runtimes[node] = {}
            speed: float = network.nodes[node]["weight"]
            for task in task_graph.nodes:
                cost: float = task_graph.nodes[task]["weight"]
                runtimes[node][task] = cost / speed
                logging.debug(
                    "Task %s on node %s has runtime %s",
                    task,
                    node,
                    runtimes[node][task],
                )

        commtimes = {}

        # every two nodes in network graph
        for src, dst in network.edges:

            # source -> destination
            commtimes[src, dst] = {}

            # destination -> source
            commtimes[dst, src] = {}

            # bandwidth of two nodes
            speed: float = network.edges[src, dst]["weight"]

            # dependence: src_task -> dst_task
            for src_task, dst_task in task_graph.edges:

                # total data transmission needed
                cost = task_graph.edges[src_task, dst_task]["weight"]

                # communication time
                commtimes[src, dst][src_task, dst_task] = cost / speed
                commtimes[dst, src][src_task, dst_task] = cost / speed
                logging.debug(
                    "Task %s on node %s to task %s on node %s has communication time %s",
                    src_task,
                    src,
                    dst_task,
                    dst,
                    commtimes[src, dst][src_task, dst_task],
                )

        return runtimes, commtimes

    def _schedule(
        self,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        runtimes: Dict[Hashable, Dict[Hashable, float]],
        commtimes: Dict[
            Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]
        ],
        schedule_order: List[Hashable],
    ) -> Dict[Hashable, List[Task]]:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            runtimes (Dict[Hashable, Dict[Hashable, float]]): A dictionary mapping nodes to a
                dictionary of tasks and their runtimes.
            commtimes (Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]): A
                dictionary mapping edges to a dictionary of task dependencies and their communication times.
            schedule_order (List[Hashable]): The order in which to schedule the tasks.

        Returns:
            Dict[Hashable, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """

        # tasks for each node
        comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}

        # tasks are scheduled
        task_schedule: Dict[Hashable, Task] = {}

        task_name: Hashable
        logging.debug("Schedule order: %s", schedule_order)

        # schedule tasks based on order
        for task_name in schedule_order:

            min_finish_time = np.inf
            best_node = None

            for node in network.nodes:  # Find the best node to run the task

                # the earliest time the task can start on specific node
                # if no predecessor, the time is 0
                max_arrival_time: float = max(  
                    [
                        0.0, * [ task_schedule[parent].end 
                            + ( commtimes[(task_schedule[parent].node, node)][(parent, task_name)] )
                            for parent in task_graph.predecessors(task_name)
                        ],
                    ]
                )

                # get the runtime on this node
                runtime = runtimes[node][task_name]

                # Get the location where the task should be inserted in the list of tasks
                idx, start_time = get_insert_loc(
                    comp_schedule[node], max_arrival_time, runtime
                )

                logging.debug(
                    "Testing task %s on node %s: start time %s, finish time %s",
                    task_name,
                    node,
                    start_time,
                    start_time + runtime,
                )

                # find the finish time
                finish_time = start_time + runtime

                # if finish time is better, renew the best_node variable
                # based on earliest finishing time
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_node = node, idx

            # runtime of best node
            new_runtime = runtimes[best_node[0]][task_name]

            # create new Task object
            task = Task(
                best_node[0], task_name, min_finish_time - new_runtime, min_finish_time
            )

            # insert task to schedule on that specific task on that node
            comp_schedule[best_node[0]].insert(best_node[1], task)
            # insert task to task scheduled
            task_schedule[task_name] = task

        return comp_schedule

    def schedule(
        self, network: nx.Graph, task_graph: nx.DiGraph
    ) -> Dict[str, List[Task]]:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[str, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """
        # get runtimes and communication times
        runtimes, commtimes = HeftScheduler.get_runtimes(network, task_graph)

        # get order of scheduling based on upward ranking
        schedule_order = heft_rank_sort(network, task_graph)

        # return the scheduled result
        return self._schedule(network, task_graph, runtimes, commtimes, schedule_order)
