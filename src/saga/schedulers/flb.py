import logging
from pprint import pformat
from queue import PriorityQueue
from typing import Dict, Optional, Tuple

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph


class FLBScheduler(Scheduler):
    """The FLB (Fast Load Balancing) scheduler.

    Source: https://doi.org/10.1109/ICPP.1999.797442
    Note: They assume homogenous comp/comm speeds, so we will scale the weights in the task graph
        by the average (but it will still perform poorly on heterogeneous networks). We also schedule
        to the fastest node whenever the original algorithm schedules to an arbitrary node.
    """

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        comp_schedule = Schedule(task_graph, network)
        scheduled_tasks: Dict[str, ScheduledTask] = {}

        if schedule is not None:
            comp_schedule = schedule.model_copy()
            scheduled_tasks = {
                t.name: t for _, tasks in schedule.items() for t in tasks
            }

        # Calculate average comm speed (exclude self-loops unless single node)
        num_nodes = len(list(network.nodes))
        non_self_edges = [
            edge
            for edge in network.edges
            if edge.source != edge.target or num_nodes == 1
        ]
        avg_comm_speed = (
            sum(edge.speed for edge in non_self_edges) / len(non_self_edges)
            if non_self_edges
            else 1.0
        )

        def getEP(task_name: str) -> str:  # pylint: disable=invalid-name
            """Get Enabling Processor (EP) of a task."""
            in_edges = task_graph.in_edges(task_name)
            enabling_edge = max(
                in_edges,
                key=lambda in_edge: scheduled_tasks[in_edge.source].end
                + in_edge.size / avg_comm_speed,
            )
            return scheduled_tasks[enabling_edge.source].node

        def getLMT(task_name: str) -> float:  # pylint: disable=invalid-name
            """Get the Last Message Arrival Time (LMT)"""
            in_edges = task_graph.in_edges(task_name)
            if not in_edges:
                return min_start_time

            return max(
                scheduled_tasks[in_edge.source].end + (in_edge.size / avg_comm_speed)
                for in_edge in in_edges
            )

        def getEMT(task_name: str, node_name: str) -> float:  # pylint: disable=invalid-name
            """Get the Effective Message Arrival Time (EMT)"""
            in_edges = task_graph.in_edges(task_name)
            if not in_edges:
                return min_start_time

            return max(
                scheduled_tasks[in_edge.source].end
                + (
                    in_edge.size
                    / network.get_edge(
                        scheduled_tasks[in_edge.source].node, node_name
                    ).speed
                )
                for in_edge in in_edges
            )

        def getPRT(node_name: str) -> float:  # pylint: disable=invalid-name
            tasks = comp_schedule[node_name]
            return min_start_time if not tasks else tasks[-1].end

        def getEST(task_name: str, node_name: str) -> float:  # pylint: disable=invalid-name
            return max(getPRT(node_name), getEMT(task_name, node_name))

        non_ep_tasks: PriorityQueue[Tuple[float, str]] = PriorityQueue()
        emt_ep_tasks: Dict[str, PriorityQueue[Tuple[float, str]]] = {
            node.name: PriorityQueue() for node in network.nodes
        }
        lmt_ep_tasks: Dict[str, PriorityQueue[Tuple[float, str]]] = {
            node.name: PriorityQueue() for node in network.nodes
        }
        all_procs: PriorityQueue[Tuple[float, str]] = PriorityQueue()
        active_procs: PriorityQueue[Tuple[float, str]] = PriorityQueue()

        for _task in task_graph.tasks:
            if (
                _task.name not in scheduled_tasks
                and task_graph.in_degree(_task.name) == 0
            ):
                non_ep_tasks.put((min_start_time, _task.name))

        for _node in network.nodes:
            all_procs.put((min_start_time, _node.name))

        def schedule_task() -> Tuple[str, str]:
            # get head of active_procs without removing
            _, proc1 = active_procs.queue[0] if active_procs.queue else (0, None)
            task1 = None
            if proc1 is not None:
                _, task1 = (
                    emt_ep_tasks[proc1].queue[0]
                    if emt_ep_tasks[proc1].queue
                    else (0, None)
                )
            _, proc2 = all_procs.queue[0] if all_procs.queue else (0, None)
            _, task2 = non_ep_tasks.queue[0] if non_ep_tasks.queue else (0, None)

            est_t1_p1 = (
                getEST(task1, proc1)
                if proc1 is not None and task1 is not None
                else float("inf")
            )
            est_t2_p2 = (
                getEST(task2, proc2)
                if proc2 is not None and task2 is not None
                else float("inf")
            )
            if est_t1_p1 == float("inf") and est_t2_p2 == float("inf"):
                logging.debug("active_procs: %s", pformat(active_procs.queue))
                logging.debug("all_procs: %s", pformat(all_procs.queue))
                logging.debug("non_ep_tasks: %s", pformat(non_ep_tasks.queue))
                logging.debug(
                    "emt_ep_tasks: %s",
                    pformat(
                        {
                            node: pformat(emt_ep_tasks[node].queue)
                            for node in emt_ep_tasks
                        }
                    ),
                )
                logging.debug("schedule: %s", pformat(comp_schedule))

                raise RuntimeError(
                    f"No tasks to schedule. proc1={proc1}, task1={task1}, proc2={proc2}, task2={task2}"
                )

            if est_t1_p1 <= est_t2_p2:
                if task1 is None or proc1 is None:
                    raise RuntimeError(
                        "Inconsistent state in active_procs or emt_ep_tasks."
                    )  # Should not happen
                task_obj = task_graph.get_task(task1)
                node_obj = network.get_node(proc1)
                new_task = ScheduledTask(
                    node=proc1,
                    name=task1,
                    start=est_t1_p1,
                    end=est_t1_p1 + task_obj.cost / node_obj.speed,
                )
                comp_schedule.add_task(new_task)
                scheduled_tasks[task1] = new_task

                assert active_procs.get()[1] == proc1
                assert emt_ep_tasks[proc1].get()[1] == task1
                lmt_ep_tasks[proc1].queue = [
                    (priority, task)
                    for priority, task in lmt_ep_tasks[proc1].queue
                    if task != task1
                ]
                return task1, proc1
            else:
                if task2 is None or proc2 is None:
                    raise RuntimeError(
                        "Inconsistent state in all_procs or non_ep_tasks."
                    )  # Should not happen
                task_obj = task_graph.get_task(task2)
                node_obj = network.get_node(proc2)
                new_task = ScheduledTask(
                    node=proc2,
                    name=task2,
                    start=est_t2_p2,
                    end=est_t2_p2 + task_obj.cost / node_obj.speed,
                )
                comp_schedule.add_task(new_task)
                scheduled_tasks[task2] = new_task

                assert all_procs.get()[1] == proc2
                all_procs.put((getPRT(proc2), proc2))
                assert non_ep_tasks.get()[1] == task2
                return task2, proc2

        def update_task_lists(task_name: str, proc: str):
            while True:
                _, task = (
                    lmt_ep_tasks[proc].queue[0]
                    if lmt_ep_tasks[proc].queue
                    else (0, None)
                )
                if task is None:
                    break
                if getLMT(task) >= getPRT(proc):
                    break

                assert lmt_ep_tasks[proc].get()[1] == task
                emt_ep_tasks[proc].queue = [
                    (priority, _task)
                    for priority, _task in emt_ep_tasks[proc].queue
                    if _task != task
                ]
                non_ep_tasks.put((getLMT(task), task))

        def update_proc_lists(task_name: str, proc: str):
            _, task = (
                emt_ep_tasks[proc].queue[0] if emt_ep_tasks[proc].queue else (0, None)
            )
            if task is None:
                active_procs.queue = [
                    (priority, _proc)
                    for priority, _proc in active_procs.queue
                    if _proc != proc
                ]
            else:
                active_procs.queue = [
                    (priority, _proc)
                    for priority, _proc in active_procs.queue
                    if _proc != proc
                ]
                active_procs.put((getEST(task, proc), proc))

        def update_ready_tasks(task_name: str, proc: str):
            for out_edge in task_graph.out_edges(task_name):
                succ = out_edge.target
                if succ in scheduled_tasks:
                    continue

                if not all(
                    in_edge.source in scheduled_tasks
                    for in_edge in task_graph.in_edges(succ)
                ):
                    continue

                enabling_proc = getEP(succ)
                lmt = getLMT(succ)
                emt = getEMT(succ, enabling_proc)
                if lmt < getPRT(enabling_proc):
                    non_ep_tasks.put((lmt, succ))
                else:
                    _, head_proc = (
                        emt_ep_tasks[enabling_proc].queue[0]
                        if emt_ep_tasks[enabling_proc].queue
                        else (0, None)
                    )
                    if head_proc is None:
                        active_procs.put((getEST(succ, enabling_proc), enabling_proc))
                    else:
                        head_task = emt_ep_tasks[enabling_proc].queue[0][1]
                        _emt = getEMT(head_task, enabling_proc)
                        if emt < _emt:
                            new_priority = max(emt, getPRT(enabling_proc))
                            active_procs.queue = [
                                (priority, _proc)
                                for priority, _proc in active_procs.queue
                                if _proc != enabling_proc
                            ]
                            active_procs.put((new_priority, enabling_proc))
                    emt_ep_tasks[enabling_proc].put((emt, succ))
                    lmt_ep_tasks[enabling_proc].put((lmt, succ))

        num_tasks = len(list(task_graph.tasks))
        while len(scheduled_tasks) < num_tasks:
            task, proc = schedule_task()
            update_task_lists(task, proc)
            update_proc_lists(task, proc)
            update_ready_tasks(task, proc)

        return comp_schedule
