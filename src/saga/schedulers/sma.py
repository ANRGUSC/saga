import pathlib
from typing import Dict, Hashable, List, Tuple, Set
import random
import math
import networkx as nx
import numpy as np


from ..scheduler import Scheduler, Task


thisdir = pathlib.Path(__file__).resolve().parent

class SimulatedAnnealingScheduler(Scheduler):
    def __init__(self, initial_temperature: float, cooling_rate: float, stopping_temperature: float, num_iterations: int):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.stopping_temperature = stopping_temperature
        self.num_iterations = num_iterations
        self.task_graph = None
        self.commtimes = None

    @staticmethod
    def get_runtimes(
        network: nx.Graph, task_graph: nx.DiGraph
    ) -> Tuple[
        Dict[Hashable, Dict[Hashable, float]],
        Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]
    ]:
        """Get the expected runtimes of all tasks on all nodes."""
        runtimes = {}
        for node in network.nodes:
            runtimes[node] = {}
            speed = network.nodes[node]["weight"]
            for task in task_graph.nodes:
                cost = task_graph.nodes[task]["weight"]
                runtimes[node][task] = cost / speed

        commtimes = {}
        for src, dst in network.edges:
            commtimes[src, dst] = {}
            commtimes[dst, src] = {}
            speed = network.edges[src, dst]["weight"]
            for src_task, dst_task in task_graph.edges:
                cost = task_graph.edges[src_task, dst_task]["weight"]
                commtimes[src, dst][src_task, dst_task] = cost / speed
                commtimes[dst, src][src_task, dst_task] = cost / speed

        return runtimes, commtimes

    def calculate_cost(
        self,
        schedule: Dict[Hashable, List[Task]],
        task_graph: nx.DiGraph,
        commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]
    ) -> float:
        task_end_times = {}
        node_available_times = {node: 0 for node in schedule.keys()}
        max_finish_time = 0

        for node in nx.topological_sort(task_graph):
            if node in ['__src__', '__dst__']:
                continue

            assigned_node = self.get_task_node(node, schedule)
            task = next(t for t in schedule[assigned_node] if t.name == node)

            start_time = node_available_times[assigned_node]
            for predecessor in task_graph.predecessors(node):
                if predecessor in ['__src__', '__dst__']:
                    continue
                pred_node = self.get_task_node(predecessor, schedule)
                pred_end_time = task_end_times.get(predecessor, 0)
                comm_time = commtimes.get((pred_node, assigned_node), {}).get((predecessor, node), 0)
                start_time = max(start_time, pred_end_time + comm_time)

            end_time = start_time + (task.end - task.start)
            task.start = start_time
            task.end = end_time
            task_end_times[node] = end_time
            node_available_times[assigned_node] = end_time
            max_finish_time = max(max_finish_time, end_time)

        return max_finish_time

    def get_task_node(self, task_name: str, schedule: Dict[Hashable, List[Task]]) -> Hashable:
        for node, tasks in schedule.items():
            for task in tasks:
                if task.name == task_name:
                    return node
        raise ValueError(f"Task {task_name} not found in the schedule")

    def initial_solution(self, schedule: Dict[Hashable, List[Task]], runtimes: Dict[Hashable, Dict[Hashable, float]], network, task_graph) -> Dict[Hashable, List[Task]]:
        """Generate the initial solution by assigning all tasks to a single node."""
        task_list = list(nx.topological_sort(task_graph))
        task_end_times = {}

        if network.nodes:
            assigned_node = random.choice(list(network.nodes))
        else:
            raise ValueError("Network has no nodes to assign tasks.")

        schedule = {node: [] for node in network.nodes}
        node_available_time = 0

        for task in task_list:
            if task in ['__src__', '__dst__']:
                continue
            
            start_time = node_available_time
            for parent in task_graph.predecessors(task):
                if parent not in ['__src__', '__dst__']:
                    comm_time = self.commtimes.get((assigned_node, assigned_node), {}).get((parent, task), 0)
                    start_time = max(start_time, task_end_times.get(parent, 0) + comm_time)
            
            end_time = start_time + runtimes[assigned_node][task]
            
            new_task = Task(node=assigned_node, name=task, start=start_time, end=end_time)
            schedule[assigned_node].append(new_task)
            task_end_times[task] = end_time
            node_available_time = end_time

        return schedule

    def generate_neighbor(self, schedule: Dict[Hashable, List[Task]], runtimes: Dict[Hashable, Dict[Hashable, float]], network, task_graph) -> Dict[Hashable, List[Task]]:
        nodes = list(schedule.keys())
        tasks_to_schedule = {node: {task.name for task in tasks} for node, tasks in schedule.items()}
        
        if len(nodes) == 1:
            return schedule
        node1, node2 = random.sample(nodes, 2)
        
        if schedule[node1] and not schedule[node2]:
            source_node = node1
            target_node = node2
            movable_tasks = [task for task in tasks_to_schedule[source_node] if not any(successor in tasks_to_schedule[source_node] for successor in task_graph.successors(task))]
            if movable_tasks:
                task_to_move = random.choice(movable_tasks)
                tasks_to_schedule[source_node].remove(task_to_move)
                tasks_to_schedule[target_node].add(task_to_move)
        elif schedule[node2] and not schedule[node1]:
            source_node = node2
            target_node = node1
            movable_tasks = [task for task in tasks_to_schedule[source_node] if not any(successor in tasks_to_schedule[source_node] for successor in task_graph.successors(task))]
            if movable_tasks:
                task_to_move = random.choice(movable_tasks)
                tasks_to_schedule[source_node].remove(task_to_move)
                tasks_to_schedule[target_node].add(task_to_move)
        elif schedule[node2] and schedule[node1]:
            movable_tasks1 = [task for task in tasks_to_schedule[node1] if not any(successor in tasks_to_schedule[node1] for successor in task_graph.successors(task))]
            movable_tasks2 = [task for task in tasks_to_schedule[node2] if not any(successor in tasks_to_schedule[node2] for successor in task_graph.successors(task))]
            if movable_tasks1 and movable_tasks2:
                task1 = random.choice(movable_tasks1)
                task2 = random.choice(movable_tasks2)
                tasks_to_schedule[node1].remove(task1)
                tasks_to_schedule[node2].remove(task2)
                tasks_to_schedule[node1].add(task2)
                tasks_to_schedule[node2].add(task1)
        else:
            return schedule
        
        new_schedule = {node: [] for node in schedule.keys()}
        self.recalculate_times(new_schedule, tasks_to_schedule, runtimes, task_graph, self.commtimes)
        return new_schedule

    def recalculate_times(self, schedule: Dict[Hashable, List[Task]], tasks_to_schedule: Dict[Hashable, Set[str]], 
                          runtimes: Dict[Hashable, Dict[Hashable, float]], task_graph: nx.DiGraph, 
                          commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]):
        task_end_times = {}
        node_available_times = {node: 0 for node in schedule.keys()}
        
        for node in nx.topological_sort(task_graph):
            if node in ['__src__', '__dst__']:
                continue
            
            assigned_node = next(n for n, tasks in tasks_to_schedule.items() if node in tasks)
            
            start_time = node_available_times[assigned_node]
            for predecessor in task_graph.predecessors(node):
                if predecessor in ['__src__', '__dst__']:
                    continue
                pred_node = next(n for n, tasks in schedule.items() if any(t.name == predecessor for t in tasks))
                pred_end_time = task_end_times.get(predecessor, 0)
                comm_time = commtimes.get((pred_node, assigned_node), {}).get((predecessor, node), 0)
                start_time = max(start_time, pred_end_time + comm_time)
            
            end_time = start_time + runtimes[assigned_node][node]
            
            new_task = Task(node=assigned_node, name=node, start=start_time, end=end_time)
            schedule[assigned_node].append(new_task)
            
            task_end_times[node] = end_time
            node_available_times[assigned_node] = end_time
        
        for node_tasks in schedule.values():
            node_tasks.sort(key=lambda t: t.start)

    def _schedule(
        self,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        runtimes: Dict[Hashable, Dict[Hashable, float]],
        commtimes: Dict[Tuple[Hashable, Hashable], Dict[Tuple[Hashable, Hashable], float]]
    ) -> Dict[Hashable, List[Task]]:
        """Apply simulated annealing to find the best task schedule."""
        self.task_graph = task_graph
        self.commtimes = commtimes
        current_solution = self.initial_solution({}, runtimes, network, task_graph)

        current_cost = self.calculate_cost(current_solution, task_graph, commtimes)
        best_solution = current_solution
        best_cost = current_cost

        temperature = self.initial_temperature

        while temperature > self.stopping_temperature:
            for _ in range(self.num_iterations):
                neighbor_solution = self.generate_neighbor(current_solution, runtimes, network, task_graph)
                neighbor_cost = self.calculate_cost(neighbor_solution, task_graph, commtimes)

                if neighbor_cost < current_cost or random.random() < math.exp((current_cost - neighbor_cost) / temperature):
                    current_solution = neighbor_solution
                    current_cost = neighbor_cost

                    if neighbor_cost < best_cost:
                        best_solution = neighbor_solution
                        best_cost = neighbor_cost

            temperature *= self.cooling_rate

        return best_solution

    def schedule(
        self, network: nx.Graph, task_graph: nx.DiGraph
    ) -> Dict[int, List[Task]]:
        """Schedule the tasks on the network."""
        runtimes, commtimes = self.get_runtimes(network, task_graph)
        self.commtimes = commtimes
        best_solution = self._schedule(network, task_graph, runtimes, commtimes)
        if not best_solution:
            raise ValueError("Failed to generate a valid schedule.")
        return {node: tasks for node, tasks in best_solution.items()}