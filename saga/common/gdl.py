from typing import Dict, Hashable, List, Tuple, Optional
import networkx as nx
import numpy as np

from ..base import Scheduler,Task
from ..utils.tools import check_instance_simple

class GDLScheduler(Scheduler):
    def __init__(self):
        super(GDLScheduler, self).__init__()

    def get_runtimes(self, network: nx.Graph, task_graph: nx.DiGraph) -> Tuple[Dict[Hashable, float], Dict[Tuple[Hashable, Hashable], float]]:
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

    def compute_b_level(self, task, tasks, computation_costs, communication_costs):
        successors = tasks.successors(task)
        if not successors:
            return computation_costs[task]
        else:
            return computation_costs[task] + max(
                communication_costs[(task, s)] + self.compute_b_level(s, tasks, computation_costs, communication_costs) for
                s in successors)

    def compute_est(self, task, processor, schedule, communication_costs):
        if not schedule.predecessors(task):
            return 0
        else:
            return max(
                schedule[predecessor][task]['finish_time'] + communication_costs[(predecessor, task)] for predecessor in
                schedule.predecessors(task))

    def update_priorities(self, tasks, schedule, computation_costs, communication_costs):
        for task in tasks:
            task['priority'] = self.compute_b_level(task, tasks, computation_costs, communication_costs) - self.compute_est(task, task['processor'], schedule, communication_costs)

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        # Initial empty schedule
        schedule = nx.DiGraph()

        # Get task list from task_graph
        tasks = list(task_graph.nodes)
        tasks = [Task(node=t, name=str(t)) for t in tasks]

        # Get computation and communication costs
        computation_costs, communication_costs = self.get_runtimes(network, task_graph)

        # Main scheduling loop
        while tasks:
            # Update task priorities
            self.update_priorities(tasks, schedule, computation_costs, communication_costs)

            # Select task with highest priority
            task = max(tasks, key=lambda t: t['priority'])

            # Assign task to processor with earliest finish time
            est = {p: self.compute_est(task, p, schedule, communication_costs) for p in network}
            task.node = min(est, key=est.get)

            # Add task to schedule
            schedule.add_node(task)

            # Remove task from task list
            tasks.remove(task)

        # Convert schedule to output format
        schedule_output = {}
        for task in schedule.nodes:
            if schedule.nodes[task]['processor'] not in schedule_output:
                schedule_output[schedule.nodes[task]['processor']] = []
            schedule_output[schedule.nodes[task]['processor']].append(Task(node=task, name=str(task), start=schedule.nodes[task]['start_time'], end=schedule.nodes[task]['finish_time']))

        return schedule_output
