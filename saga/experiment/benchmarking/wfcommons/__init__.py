import pathlib
import random
import tempfile
from dataclasses import dataclass
from itertools import product
from typing import List, Type

import dill as pickle
import networkx as nx
import numpy as np
from saga.experiment.pisa.changes import Change
from saga.experiment.pisa.simulated_annealing import SimulatedAnnealing

from saga.scheduler import Scheduler
from saga.schedulers.data.wfcommons import (get_networks, get_num_task_range,
                                            get_real_networks, get_real_workflows,
                                            get_workflow_task_info, recipes,
                                            trace_to_digraph)
from wfcommons.wfgen import WorkflowGenerator

def get_changes(cloud_type: str,
                recipe_name: str,
                granularity: float) -> List[Type[Change]]:
    """Returns a list of change types for the given cloud type and granularity.

    Args:
        cloud_type (str): The cloud type for the network.
        recipe_name (str): The recipe name for the workflow.
        granularity (float): The granularity of weight changes. A higher granularity means
            smaller changes between neighbor graphs

    Returns:
        List[Type[Change]]: A list of change types for the given cloud type and granularity.
    """
    real_networks = get_real_networks(cloud_type)
    min_network_nodes = min(network.number_of_nodes() for network in real_networks)
    max_network_nodes = max(network.number_of_nodes() for network in real_networks)
    min_node_weight = min(network.nodes[node]["weight"]
                          for network in real_networks
                          for node in network.nodes)
    max_node_weight = max(network.nodes[node]["weight"]
                          for network in real_networks
                          for node in network.nodes)
    task_info = get_workflow_task_info(recipe_name)

    @dataclass
    class ChangeNetworkWeight(Change):
        """Change the weight of a node in the network."""
        node: str
        weight: float

        def __str__(self) -> str:
            return f"ChangeNetworkWeight[cloud_type={cloud_type},granularity={granularity}](node={self.node}, weight={self.weight})"

        @classmethod
        def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> 'ChangeNetworkWeight':
            d_weight = (max_node_weight - min_node_weight) / granularity
            node = random.choice([node for node in network.nodes])
            weight = network.nodes[node]["weight"] + random.choice([-d_weight, d_weight])
            weight = min(max(weight, min_node_weight), max_node_weight)
            return ChangeNetworkWeight(node, weight)

        def apply(self, network: nx.Graph, task_graph: nx.DiGraph) -> None:
            network.nodes[self.node]["weight"] = self.weight

    @dataclass
    class ChangeTaskWeight(Change):
        """Change the weight of a task in the task graph."""
        task: str
        weight: float

        def __str__(self) -> str:
            return f"ChangeTaskWeight[cloud_type={cloud_type},granularity={granularity}](task={self.task}, weight={self.weight})"

        @classmethod
        def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> 'ChangeTaskWeight':
            task: str = random.choice([task for task in task_graph.nodes])
            task_type, task_id = task.rsplit("_", 1)
            min_task_weight = task_info[task_type]['runtime']['min']
            max_task_weight = task_info[task_type]['runtime']['max']

            d_weight = (max_node_weight - min_node_weight) / granularity
            weight = task_graph.nodes[task]["weight"] + random.choice([-d_weight, d_weight])
            weight = min(max(weight, min_task_weight), max_task_weight)
            return ChangeTaskWeight(task, weight)

        def apply(self, network: nx.Graph, task_graph: nx.DiGraph) -> None:
            task_graph.nodes[self.task]["weight"] = self.weight

    @dataclass
    class ChangeDependencyWeight(Change):
        """Change the weight of a dependency in the task graph."""
        task: str
        dependency: str
        weight: float

        def __str__(self) -> str:
            return f"ChangeDependencyWeight[cloud_type={cloud_type},granularity={granularity}](task={self.task}, dependency={self.dependency}, weight={self.weight})"

        @classmethod
        def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> 'ChangeDependencyWeight':
            task, child = random.choice([edge for edge in task_graph.edges])
            task_type = task.rsplit("_", 1)[0]
            child_type = child.rsplit("_", 1)[0]
            task_outputs = set(task_info[task_type]["output"].keys())
            child_inputs = [
                (input_info["min"], input_info["max"])
                for input_name, input_info in task_info[child_type]["input"].items()
                if input_name in task_outputs
            ]
            min_weight = sum(min_size for min_size, _ in child_inputs)
            max_weight = sum(max_size for _, max_size in child_inputs)
            d_weight = (max_weight - min_weight) / granularity
            weight = task_graph.edges[task, child]["weight"] + random.choice([-d_weight, d_weight])
            weight = min(max(weight, min_weight), max_weight)
            return ChangeDependencyWeight(task, child, weight)

        def apply(self, network: nx.Graph, task_graph: nx.DiGraph) -> None:
            task_graph.edges[self.task, self.dependency]["weight"] = self.weight

    @dataclass
    class AddNetworkNode(Change):
        """Add a node to the network."""
        node: str
        weight: float

        def __str__(self) -> str:
            return f"AddNetworkNode[cloud_type={cloud_type},granularity={granularity}](node={self.node}, weight={self.weight})"

        @classmethod
        def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> 'AddNetworkNode':
            if network.number_of_nodes() >= max_network_nodes:
                return None
            node_ids = [int(node) for node in network.nodes]
            # get first missing node id
            node = next(i for i in range(max_network_nodes) if i not in node_ids)
            weight = random.uniform(min_node_weight, max_node_weight)
            return AddNetworkNode(node, weight)

        def apply(self, network: nx.Graph, task_graph: nx.DiGraph) -> None:
            network.add_node(self.node, weight=self.weight)
            for node in network.nodes:
                network.add_edge(node, self.node, weight=1e9)

    @dataclass
    class RemoveNetworkNode(Change):
        """Remove a node from the network."""
        node: str

        def __str__(self) -> str:
            return f"RemoveNetworkNode[cloud_type={cloud_type},granularity={granularity}](node={self.node})"

        @classmethod
        def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> 'RemoveNetworkNode':
            if network.number_of_nodes() <= min_network_nodes:
                return None
            node = random.choice([node for node in network.nodes])
            return RemoveNetworkNode(node)

        def apply(self, network: nx.Graph, task_graph: nx.DiGraph) -> None:
            network.remove_node(self.node)

    return [
        ChangeNetworkWeight, ChangeTaskWeight,
        AddNetworkNode, RemoveNetworkNode,
        ChangeDependencyWeight
    ]

def get_smallest_task_graph(recipe_name: str) -> nx.DiGraph:
    """Returns the smallest task graph for the given recipe name.

    Args:
        recipe_name (str): The recipe name for the workflow.

    Returns:
        nx.DiGraph: The smallest task graph for the given recipe name.
    """
    min_num_tasks, _ = get_num_task_range(recipe_name)
    recipes[recipe_name](num_tasks=min_num_tasks)
    recipe = recipes[recipe_name](num_tasks=min_num_tasks)
    generator = WorkflowGenerator(recipe)
    workflow = generator.build_workflow()
    with tempfile.NamedTemporaryFile() as tmp:
        workflow.write_json(tmp.name)
        task_graph = trace_to_digraph(tmp.name)
    return task_graph

def run(output_path: pathlib.Path,
        cloud: str,
        recipe_name: str,
        ccr: float,
        schedulers: List[Scheduler],
        neighbor_granularity: float = 10,
        max_iterations: int = 1000,
        num_runs: int = 1):
    """Runs experiment.
    
    Args:
        output_path (pathlib.Path): Path to save results.
        cloud (str): The cloud type for the network.
        recipe_name (str): The recipe name for the workflow.
        ccr (float): The CCR for the network.
        schedulers (List[Scheduler]): The schedulers to run.
        neighbor_granularity (float, optional): The granularity of weight changes. A higher granularity means
            smaller changes between neighbor graphs. Defaults to 10.
        max_iterations (int, optional): The maximum number of iterations for the simulated annealing algorithm. Defaults to 1000.
        num_runs (int, optional): The number of runs to perform. Defaults to 1.
    """

    workflows = get_real_workflows(recipe_name)
    mean_task_weight = np.mean([
        workflow.nodes[node]["weight"]
        for workflow in workflows
        for node in workflow.nodes
    ])
    mean_edge_weight = np.mean([
        workflow.edges[edge]["weight"]
        for workflow in workflows
        for edge in workflow.edges
    ])

    network_speed = mean_edge_weight / (ccr * mean_task_weight) 

    all_schedulers = {
        scheduler.__class__.__name__.removesuffix("Scheduler"): scheduler
        for scheduler in schedulers
    }
    rerun_all = False
    rerun_schedulers = {"MinMin"}
    rerun_base_schedulers = set()

    for _ in range(num_runs):
        for (target_sched_name, target_scheduler), (base_sched_name, base_scheduler) in product(all_schedulers.items(), all_schedulers.items()):
            if target_sched_name == base_sched_name:
                continue
            savepath = output_path.joinpath(f'{base_sched_name}/{target_sched_name}.pkl')
            savepath.parent.mkdir(parents=True, exist_ok=True)

            if not rerun_all and savepath.exists() and target_sched_name not in rerun_schedulers and base_sched_name not in rerun_base_schedulers:
                print(f"Skipping {base_sched_name}/{target_sched_name}")
                continue

            old_energy = 0
            if savepath.exists():
                sa_old = pickle.loads(savepath.read_bytes())
                old_energy = sa_old.iterations[-1].best_energy
            print(f"Running target={target_sched_name}, base={base_sched_name}, old_energy={old_energy}")

            initial_network = get_networks(1, cloud, network_speed=network_speed)[0]
            initial_task_graph = get_smallest_task_graph(recipe_name)

            sa = SimulatedAnnealing(
                task_graph=initial_task_graph,
                network=initial_network,
                scheduler=target_scheduler,
                base_scheduler=base_scheduler,
                change_types=get_changes(
                    cloud_type=cloud,
                    recipe_name=recipe_name,
                    granularity=neighbor_granularity
                ),
                max_iterations=max_iterations
            )

            sa.run()

            print(f"base={base_sched_name}, target={target_sched_name} energy={sa.iterations[-1].best_energy}")
            if old_energy > sa.iterations[-1].best_energy:
                print(f"Skipping base={base_sched_name}, target={target_sched_name} because old energy is better (old={old_energy}, new={sa.iterations[-1].best_energy})")
            else:
                print(f"Saving base={base_sched_name}, target={target_sched_name} because new energy is better (old={old_energy}, new={sa.iterations[-1].best_energy})")
                savepath.write_bytes(pickle.dumps(sa))


def run_many(output_path: pathlib.Path,
             cloud: str,
             ccrs: List[float],
             schedulers: List[Scheduler],
             recipe_name: str = None,
             neighbor_granularity: float = 10,
             max_iterations: int = 1000,
             num_runs: int = 1):
    """Runs experiment.

    Args:
        output_path (pathlib.Path): Path to save results.
        cloud (str): The cloud type for the network.
        recipe_name (str): The recipe name for the workflow.
        ccrs (List[float]): The CCRs for the network.
        schedulers (List[Scheduler]): The schedulers to run.
        neighbor_granularity (float, optional): The granularity of weight changes. A higher granularity means
            smaller changes between neighbor graphs. Defaults to 10.
        max_iterations (int, optional): The maximum number of iterations for the simulated annealing algorithm. Defaults to 1000.
        num_runs (int, optional): The number of runs to perform. Defaults to 1.
    """
    all_recipes = [
        name
        for name in recipes.keys()
        if recipe_name is None or recipe_name == name
    ]
    for recipe_name in all_recipes:
        for ccr in ccrs:
            run(output_path=output_path.joinpath(recipe_name, f"ccr_{ccr}"),
                cloud=cloud,
                recipe_name=recipe_name,
                ccr=ccr,
                schedulers=schedulers,
                neighbor_granularity=neighbor_granularity,
                max_iterations=max_iterations,
                num_runs=num_runs)