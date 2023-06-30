from typing import Dict, Hashable, List, Tuple, Optional
import networkx as nx
import numpy as np

from ..base import Scheduler, Task
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

    def compute_b_level(self, task, task_graph, computation_costs, communication_costs):
        successors = task_graph.successors(task)
        if not list(successors):
            return computation_costs[task]
        else:
            values = [communication_costs.get((task, s), 0) + self.compute_b_level(s, task_graph, computation_costs, communication_costs) for s in successors]
            return computation_costs[task] + (max(values) if values else 0)

    def compute_est(self, task, processor, schedule, communication_costs):
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

    def update_priorities(self, task_graph, schedule, computation_costs, communication_costs, task_priorities, processor_for_task):
        for task in task_graph.nodes:
            processor = processor_for_task[task] 
            task_priorities[task] = self.compute_b_level(task, task_graph, computation_costs, communication_costs) - self.compute_est(task, processor, schedule, communication_costs)

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        check_instance_simple(network, task_graph)
        schedule = nx.DiGraph()

        tasks = list(task_graph.nodes)
        computation_costs, communication_costs = self.get_runtimes(network, task_graph)
        task_priorities = {task: 0 for task in tasks}
        processor_for_task = {task: None for task in tasks}

         # Main scheduling loop
        while tasks:
            # Update task priorities
            self.update_priorities(task_graph, schedule, computation_costs, communication_costs, task_priorities, processor_for_task)

            # Select task with highest priority
            task = max(tasks, key=lambda t: task_priorities[t])

            # Assign task to processor with earliest finish time
            est = {p: self.compute_est(task, p, schedule, communication_costs) for p in network}
            chosen_processor = min(est, key=est.get)
            processor_for_task[task] = chosen_processor

            # Calculate start and finish times
            if task in schedule:
                start_time = max(est[chosen_processor], max((schedule.nodes[predecessor]['finish_time'] for predecessor in task_graph.predecessors(task) if predecessor in schedule), default=0))
            else:
                start_time = est[chosen_processor]

            finish_time = start_time + computation_costs[task]

            # Add task to schedule
            schedule.add_node(task, start_time=start_time, finish_time=finish_time)

            # Remove task from task list
            tasks.remove(task)

        # Convert schedule to output format
        schedule_output = {}
        for task in schedule.nodes:
            if processor_for_task[task] not in schedule_output:
                schedule_output[processor_for_task[task]] = []
            schedule_output[processor_for_task[task]].append(Task(node=task, name=str(task), start=schedule.nodes[task]['start_time'], end=schedule.nodes[task]['finish_time']))

        return schedule_output


