from typing import Dict, Hashable, List, Tuple

import networkx as nx

from ..base import Scheduler, Task


class GDLScheduler(Scheduler):
    """Earliest Task First scheduler"""

    def get_runtimes(self,
                     network: nx.Graph,
                     task_graph: nx.DiGraph) -> Tuple[Dict[Hashable, float],
                                                      Dict[Tuple[Hashable, Hashable], float]]:
        """Returns the computation and communication costs of the tasks

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Tuple[Dict[Hashable, float], Dict[Tuple[Hashable, Hashable], float]]:
                The computation and communication costs of the tasks.
        """
        computation_costs = {}
        communication_costs = {}
        for node in network.nodes:
            speed: float = network.nodes[node]["weight"]
            for task in task_graph.nodes:
                cost: float = task_graph.nodes[task]["weight"]
                computation_costs[task] = cost / speed

        for src, dst in network.edges:
            speed: float = network.edges[src, dst]["weight"]
            for src_task, dst_task in task_graph.edges:
                cost = task_graph.edges[src_task, dst_task]["weight"]
                communication_costs[(src_task, dst_task)] = cost / speed

        return computation_costs, communication_costs

    def compute_b_level(self,
                        task: Hashable,
                        task_graph: nx.DiGraph,
                        computation_costs: Dict[Hashable, float],
                        communication_costs: Dict[Tuple[Hashable, Hashable], float]) -> float:
        """Computes the b-level of a task

        Args:
            task (Hashable): The task.
            task_graph (nx.DiGraph): The task graph.
            computation_costs (Dict[Hashable, float]): The computation costs of the tasks.
            communication_costs (Dict[Tuple[Hashable, Hashable], float]): The communication costs of the tasks.

        Returns:
            float: The b-level of the task.
        """
        successors = task_graph.successors(task)
        if not list(successors):
            return computation_costs[task]
        else:
            values = [
                (communication_costs.get((task, s), 0) +
                 self.compute_b_level(s, task_graph, computation_costs, communication_costs))
                for s in successors
            ]
            return computation_costs[task] + (max(values) if values else 0)

    def compute_est(self,
                    task: Hashable,
                    schedule: nx.DiGraph,
                    communication_costs: Dict[Tuple[Hashable, Hashable], float]) -> float:
        """Computes the earliest start time of a task

        Args:
            task (Hashable): The task.
            schedule (nx.DiGraph): The schedule.
            communication_costs (Dict[Tuple[Hashable, Hashable], float]): The communication costs of the tasks.

        Returns:
            float: The earliest start time of the task.
        """
        try:
            predecessors = schedule.predecessors(task)
        except nx.NetworkXError:
            return 0

        if not list(predecessors):
            return 0
        else:
            return max(
                schedule[predecessor][task]['finish_time'] + communication_costs.get((predecessor, task), 0)
                for predecessor in predecessors
            )

    def update_priorities(self,
                          task_graph: nx.DiGraph,
                          schedule: nx.DiGraph,
                          computation_costs: Dict[Hashable, float],
                          communication_costs: Dict[Tuple[Hashable, Hashable], float],
                          task_priorities: Dict[Hashable, float]) -> None:
        """Updates the priorities of the tasks

        Args:
            task_graph (nx.DiGraph): The task graph.
            schedule (nx.DiGraph): The schedule.
            computation_costs (Dict[Hashable, float]): The computation costs of the tasks.
            communication_costs (Dict[Tuple[Hashable, Hashable], float]): The communication costs of the tasks.
            task_priorities (Dict[Hashable, float]): The priorities of the tasks.
        """
        for task in task_graph.nodes:
            task_priorities[task] = (
                self.compute_b_level(task, task_graph, computation_costs, communication_costs) -
                self.compute_est(task, schedule, communication_costs)
            )

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:

        schedule = nx.DiGraph()

        tasks = list(task_graph.nodes)
        computation_costs, communication_costs = self.get_runtimes(network, task_graph)
        task_priorities = {task: 0 for task in tasks}
        processor_for_task = {task: None for task in tasks}

         # Main scheduling loop
        while tasks:
            # Update task priorities
            self.update_priorities(task_graph, schedule, computation_costs, communication_costs, task_priorities)

            # Select task with highest priority
            task = max(tasks, key=lambda t: task_priorities[t])

            # Assign task to processor with earliest finish time
            est = {p: self.compute_est(task, schedule, communication_costs) for p in network}
            chosen_processor = min(est, key=est.get)
            processor_for_task[task] = chosen_processor

            # Calculate start and finish times
            if task in schedule:
                start_time = max(
                    est[chosen_processor], max(
                        (schedule.nodes[predecessor]['finish_time']
                        for predecessor in task_graph.predecessors(task)
                        if predecessor in schedule), default=0
                    )
                )
            else:
                start_time = est[chosen_processor]

            finish_time = start_time + computation_costs[task]

            # Add task to schedule
            schedule.add_node(task, start_time=start_time, finish_time=finish_time)

            # Remove task from task list
            tasks.remove(task)

        # Convert schedule to output format
        schedule_output: Dict[Hashable, List[Task]] = {}
        for task in schedule.nodes:
            if processor_for_task[task] not in schedule_output:
                schedule_output[processor_for_task[task]] = []
            new_task = Task(
                node=task, name=str(task),
                start=schedule.nodes[task]['start_time'],
                end=schedule.nodes[task]['finish_time']
            )
            schedule_output[processor_for_task[task]].append(new_task)

        return schedule_output
