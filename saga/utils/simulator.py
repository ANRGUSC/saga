from typing import Dict, Hashable, List, Tuple

import networkx as nx

from saga.scheduler import Task
from saga.utils.random_variable import RandomVariable


class Simulator:
    """Simulates stochastic task graph execution over a network."""
    def __init__(self, network: nx.Graph, task_graph: nx.DiGraph, schedule: Dict[str, List[Task]]):
        """Initializes the simulator.

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.
            schedule (Dict[str, List[Task]]): The schedule.
        """
        self.network = network
        self.task_graph = task_graph
        self.schedule = schedule
        self.tasks = {task.name: task for node, tasks in schedule.items() for task in tasks}

        # get topological rank of tasks
        levels = {}
        for task in nx.topological_sort(self.task_graph):
            levels[task] = max([0] + [levels[parent] + 1 for parent in self.task_graph.predecessors(task)])

        self.schedule_order = sorted(
            self.task_graph.nodes,
            key=lambda task: (levels[task], self.tasks[task].node, levels[task])
        )

    def sample_instances(self, num_instances: int) -> List[Tuple[nx.Graph, nx.DiGraph]]:
        """Samples a network and task graph instance.

        Args:
            num_instances (int): The number of instances to sample.

        Returns:
            Tuple[nx.Graph, nx.DiGraph]: A tuple of the network and task graph instance.
        """
        node_speeds = {}
        for node in self.network.nodes:
            speed: RandomVariable = self.network.nodes[node]["weight"]
            node_speeds[node] = speed.sample(num_samples=num_instances)
        for edge in self.network.edges:
            speed: RandomVariable = self.network.edges[edge]["weight"]
            node_speeds[edge] = speed.sample(num_samples=num_instances)

        task_costs = {}
        for task in self.task_graph.nodes:
            cost: RandomVariable = self.task_graph.nodes[task]["weight"]
            task_costs[task] = cost.sample(num_samples=num_instances)
        for edge in self.task_graph.edges:
            cost: RandomVariable = self.task_graph.edges[edge]["weight"]
            task_costs[edge] = cost.sample(num_samples=num_instances)

        instances = []
        for i in range(num_instances):
            network = nx.Graph()
            for node in self.network.nodes:
                network.add_node(node, weight=node_speeds[node][i])
            for edge in self.network.edges:
                network.add_edge(edge[0], edge[1], weight=node_speeds[edge][i])

            task_graph = nx.DiGraph()
            for task in self.task_graph.nodes:
                task_graph.add_node(task, weight=task_costs[task][i])
            for edge in self.task_graph.edges:
                task_graph.add_edge(edge[0], edge[1], weight=task_costs[edge][i])

            instances.append((network, task_graph))

        return instances

    def run(self, num_simulations: int = 1) -> List[Dict[Hashable, List[Task]]]:
        """Runs the simulation.

        Returns:
            Dict[str, List[Task]]: A dictionary mapping nodes to a list of tasks executed on the node.
        """
        instances = self.sample_instances(num_instances=num_simulations)
        results = []

        for network, task_graph in instances:
            schedule: Dict[Hashable, List[Task]] = {}
            for task_name in self.schedule_order:
                runtime = task_graph.nodes[task_name]["weight"] / network.nodes[self.tasks[task_name].node]["weight"]
                parent_arrival_times = [
                    schedule[self.tasks[parent].node][-1].end + (
                        task_graph.edges[(parent, task_name)]["weight"] /
                        network.edges[(self.tasks[parent].node, self.tasks[task_name].node)]["weight"]
                    )
                    for parent in task_graph.predecessors(task_name)
                ]
                max_parent_arrival_time = max(parent_arrival_times) if parent_arrival_times else 0
                if len(schedule.get(self.tasks[task_name].node, [])) == 0:
                    schedule[self.tasks[task_name].node] = []
                    start_time = max_parent_arrival_time
                else:
                    start_time = max(
                        schedule[self.tasks[task_name].node][-1].end,
                        max_parent_arrival_time
                    )
                task = Task(
                    name=task_name,
                    node=self.tasks[task_name].node,
                    start=start_time,
                    end=start_time + runtime
                )
                schedule[self.tasks[task_name].node].append(task)
            results.append(schedule)

        return results
