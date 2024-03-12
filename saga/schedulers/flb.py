import json
import logging
from pprint import pformat
from queue import PriorityQueue
from typing import Dict, Hashable, List, Tuple

import networkx as nx
from networkx import DiGraph, Graph

from saga.scheduler import Task

from ..scheduler import Scheduler, Task


class FLBScheduler(Scheduler):
    """The FLB (Fast Load Balancing) scheduler.

    Source: https://doi.org/10.1109/ICPP.1999.797442
    Note: They assume homogenous comp/comm speeds, so we will scale the weights in the task graph
        by the average (but it will still perform poorly on heterogeneous networks). We also schedule
        to the fastest node whenever the original algorithm schedules to an arbitrary node.
    """
    def schedule(self, network: Graph, task_graph: DiGraph) -> Dict[Hashable, List[Task]]:
        network = network.copy()
        task_graph = task_graph.copy()

        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        scheduled_tasks: Dict[Hashable, Task] = {}

        avg_comm_speed = sum(
            network.edges[edge]['weight'] for edge in network.edges
            if edge[0] != edge[1] or len(network.nodes) == 1
        ) / len(network.edges)

        fastest_node = max(network.nodes, key=lambda node: network.nodes[node]['weight'])

        def getEP(task: Hashable) -> Hashable: # pylint: disable=invalid-name
            """Get Enabling Processor (EP) of a task.

            The enabling processor of a ready task t, EP (t) is the processor from which
            the last message arrives.

            NOTE: Because this algorithm assumes homogenous comp/comm speeds, the EP is
            *not* the processor from which the last message actually arrives, but rather
            the processor from which the last message *would* arrive if the network was
            homogenous (with comm speed equal to the average comm speed).

            Args:
                task (Hashable): The task.

            Returns:
                Hashable: The enabling processor of the task.
            """
            enabling_task = max(
                task_graph.predecessors(task),
                key=lambda pred: scheduled_tasks[pred].end + task_graph.edges[pred, task]['weight'] / avg_comm_speed
            )
            return scheduled_tasks[enabling_task].node


        def getLMT(task: Hashable) -> float: # pylint: disable=invalid-name
            """Get the Last Message Arrival Time (LMT)

            NOTE: Because this original algorithm assumes homogenous comp/comm speeds, we will
            modify it to use the *actual* communication time between the tasks, rather than
            assume they are all equal to the average communication time.

            Args:
                task (Hashable): The task.

            Returns:
                float: The LMT of the task.
            """
            if task_graph.in_degree(task) == 0:
                return 0

            return max(
                scheduled_tasks[pred].end + (
                    task_graph.edges[pred, task]['weight'] /
                    avg_comm_speed
                )
                for pred in task_graph.predecessors(task)
            )

        def getEMT(task: Hashable, node: Hashable) -> float: # pylint: disable=invalid-name
            """Get the Effective Message Arrival Time (EMT)

            Since our networks have 0 comm delay between nodes and we extend for
            heterogeneous networks, this is just getLMT(task).

            Args:
                task (Hashable): The task.
                node (Hashable): The node.

            Returns:
                float: The EMT of the task.
            """
            if task_graph.in_degree(task) == 0:
                return 0

            return max(
                scheduled_tasks[pred].end + (
                    task_graph.edges[pred, task]['weight'] /
                    network.edges[scheduled_tasks[pred].node, node]['weight']
                )
                for pred in task_graph.predecessors(task)
            )

        def getPRT(node: Hashable) -> float: # pylint: disable=invalid-name
            return 0 if not schedule[node] else schedule[node][-1].end

        def getEST(task: Hashable, node: Hashable) -> float: # pylint: disable=invalid-name
            return max(getPRT(node), getEMT(task, node))

        non_ep_tasks = PriorityQueue()
        emt_ep_tasks = {
            node: PriorityQueue()
            for node in network.nodes
        }
        lmt_ep_tasks = {
            node: PriorityQueue()
            for node in network.nodes
        }
        all_procs = PriorityQueue()
        active_procs = PriorityQueue()

        for task in task_graph.nodes:
            if task_graph.in_degree(task) == 0:
                non_ep_tasks.put((0, task))

        for node in network.nodes:
            all_procs.put((0, node))

        def schedule_task() -> Tuple[Hashable, Hashable]:
            # get head of active_procs without removing
            _, proc1 = active_procs.queue[0] if active_procs.queue else (0, None)
            task1 = None
            if proc1 is not None:
                _, task1 = emt_ep_tasks[proc1].queue[0] if emt_ep_tasks[proc1].queue else (0, None)
            _, proc2 = all_procs.queue[0] if all_procs.queue else (0, None)
            _, task2 = non_ep_tasks.queue[0] if non_ep_tasks.queue else (0, None)

            est_t1_p1 = getEST(task1, proc1) if proc1 is not None and task1 is not None else float('inf')
            est_t2_p2 = getEST(task2, proc2) if proc2 is not None and task2 is not None else float('inf')
            if est_t1_p1 == float('inf') and est_t2_p2 == float('inf'):
                # NOTE: This should never happen. If it does, it means that there is a bug in the algorithm.
                # log queue values
                logging.debug("active_procs: %s", pformat(active_procs.queue))
                logging.debug("all_procs: %s", pformat(all_procs.queue))
                logging.debug("non_ep_tasks: %s", pformat(non_ep_tasks.queue))
                logging.debug("emt_ep_tasks: %s", pformat({node: pformat(emt_ep_tasks[node].queue) for node in emt_ep_tasks}))
                logging.debug("schedule: %s", pformat(schedule))
                logging.debug("task_graph: %s", json.dumps(nx.readwrite.json_graph.node_link_data(task_graph)))
                logging.debug("network: %s", json.dumps(nx.readwrite.json_graph.node_link_data(network)))


                raise RuntimeError(f"No tasks to schedule. proc1={proc1}, task1={task1}, proc2={proc2}, task2={task2}")
            if est_t1_p1 <= est_t2_p2:
                new_task = Task(
                    node=proc1,
                    name=task1,
                    start=est_t1_p1,
                    end=est_t1_p1 + task_graph.nodes[task1]['weight'] / network.nodes[proc1]['weight']
                )
                schedule[proc1].append(new_task)
                scheduled_tasks[task1] = new_task

                assert active_procs.get()[1] == proc1
                assert emt_ep_tasks[proc1].get()[1] == task1 # dequeue task from emt_ep_tasks[p1]
                lmt_ep_tasks[proc1].queue = [ # remove task1 from lmt_ep_tasks[p1]
                    (priority, task) for priority, task in lmt_ep_tasks[proc1].queue
                    if task != task1
                ]
                return task1, proc1
            else:
                # schedule task t2 on processor p2
                new_task = Task(
                    node=proc2,
                    name=task2,
                    start=est_t2_p2,
                    end=est_t2_p2 + task_graph.nodes[task2]['weight'] / network.nodes[proc2]['weight']
                )
                schedule[proc2].append(new_task)
                scheduled_tasks[task2] = new_task

                assert all_procs.get()[1] == proc2 # dequeue proc2 from all_procs

                # NOTE: this is not in the original algorithm, but it seems necessary
                all_procs.put((getPRT(proc2), proc2)) # add proc2 with new priority PRT

                assert non_ep_tasks.get()[1] == task2 # dequeue task from non_ep_tasks
                return task2, proc2

        def update_task_lists(task: Hashable, proc: Hashable):
            while True:
                _, task = lmt_ep_tasks[proc].queue[0] if lmt_ep_tasks[proc].queue else (0, None)
                if task is None:
                    break
                if getLMT(task) >= getPRT(proc): # last message arrival time of t >= processor ready time
                    break

                assert lmt_ep_tasks[proc].get()[1] == task # dequeue task from lmt_ep_tasks[p]
                emt_ep_tasks[proc].queue = [ # remove task from emt_ep_tasks[p]
                    (priority, _task) for priority, _task in emt_ep_tasks[proc].queue
                    if _task != task
                ]
                non_ep_tasks.put((getLMT(task), task)) # enqueue task in non_ep_tasks

        def update_proc_lists(task: Hashable, proc: Hashable):
            _, task = emt_ep_tasks[proc].queue[0] if emt_ep_tasks[proc].queue else (0, None)
            if task is None:
                # remove proc from active_procs
                active_procs.queue = [
                    (priority, _proc) for priority, _proc in active_procs.queue
                    if _proc != proc
                ]
            else:
                # remove proc from active_procs and add back with priority est
                active_procs.queue = [
                    (priority, _proc) for priority, _proc in active_procs.queue
                    if _proc != proc
                ]
                active_procs.put((getEST(task, proc), proc))

        def update_ready_tasks(task: Hashable, proc: Hashable):
            for succ in task_graph.successors(task):
                if succ in scheduled_tasks: # not in original algorithm, is it necessary?
                    continue

                if not all(pred in scheduled_tasks for pred in task_graph.predecessors(succ)):
                    continue

                enabling_proc = getEP(succ) # get enabling processor of succ
                lmt = getLMT(succ) # get last message arrival time of succ
                emt = getEMT(succ, enabling_proc) # get effective message arrival time of succ on enabling processor
                if lmt < getPRT(enabling_proc): # if lmt < processor ready time
                    # task perhaps should not execute on ep, since this indicates that the ep is not ready
                    # when the task is ready to execute
                    # enqueue succ with priority lmt in non_ep_tasks
                    non_ep_tasks.put((lmt, succ))
                else:
                    _, head_proc = emt_ep_tasks[enabling_proc].queue[0] if emt_ep_tasks[enabling_proc].queue else (0, None)
                    if head_proc is None:
                        # enqueue ep with priority est in active_procs
                        active_procs.put((getEST(succ, enabling_proc), enabling_proc))
                    else:
                        # get next task to execute on ep (according to current priorities)
                        head_task = emt_ep_tasks[enabling_proc].queue[0][1]
                        # get the emt of this task
                        _emt = getEMT(head_task, enabling_proc)
                        if emt < _emt:
                            # succ should execute before head_task on ep because it becomes ready before head_task
                            # update ep with priority max(EMT, prt) in active_procs
                            #   first, remove ep from active_procs
                            new_priority = max(emt, getPRT(enabling_proc))
                            active_procs.queue = [
                                (priority, _proc) for priority, _proc in active_procs.queue
                                if _proc != enabling_proc
                            ]
                            active_procs.put((new_priority, enabling_proc))
                    # enqueue succ with priority emt in emt_ep_tasks[ep]
                    emt_ep_tasks[enabling_proc].put((emt, succ))
                    # enqueue succ with priority lmt in lmt_ep_tasks[ep]
                    lmt_ep_tasks[enabling_proc].put((lmt, succ))

        while len(scheduled_tasks) < len(task_graph.nodes):
            task, proc = schedule_task()
            update_task_lists(task, proc)
            update_proc_lists(task, proc)
            update_ready_tasks(task, proc)

        return schedule

