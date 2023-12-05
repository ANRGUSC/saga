import pathlib
import random
import tempfile
from dataclasses import dataclass
from itertools import product
from typing import List, Type

import dill as pickle
import networkx as nx
import numpy as np
from changes import Change
from simulated_annealing import SimulatedAnnealing

from saga.schedulers import (CpopScheduler, DuplexScheduler,
                             FastestNodeScheduler, HeftScheduler,
                             MaxMinScheduler, MinMinScheduler, WBAScheduler)
from saga.schedulers.data.wfcommons import (get_networks, get_num_task_range,
                                            get_real_networks,
                                            get_workflow_task_info, recipes,
                                            trace_to_digraph)
from wfcommons.wfgen import WorkflowGenerator

thisdir = pathlib.Path(__file__).parent.absolute()
random.seed(1)
np.random.seed(1)


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
    class AddNetworkNode(Change):
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

    return [ChangeNetworkWeight, ChangeTaskWeight, AddNetworkNode, RemoveNetworkNode]

def get_smallest_task_graph(recipe_name: str):
    min_num_tasks, _ = get_num_task_range(recipe_name)
    recipes[recipe_name](num_tasks=min_num_tasks)
    recipe = recipes[recipe_name](num_tasks=min_num_tasks)
    generator = WorkflowGenerator(recipe)
    workflow = generator.build_workflow()
    with tempfile.NamedTemporaryFile() as tmp:
        workflow.write_json(tmp.name)
        task_graph = trace_to_digraph(tmp.name)
    return task_graph

def run(output_path: pathlib.Path):
    cloud = "chameleon"
    recipe_name = "blast"

    schedulers = {
        "HEFT": HeftScheduler(),
        "CPoP": CpopScheduler(),
        "MinMin": MinMinScheduler(),
        "MaxMin": MaxMinScheduler(),
        "Duplex": DuplexScheduler(),
        "FastestNode": FastestNodeScheduler(),
        "WBA": WBAScheduler(),
    }

    for (target_sched_name, target_scheduler), (base_sched_name, base_scheduler) in product(schedulers.items(), schedulers.items()):
        if target_sched_name == base_sched_name:
            continue
        savepath = output_path.joinpath(f'{base_sched_name}/{target_sched_name}.pkl')
        savepath.parent.mkdir(parents=True, exist_ok=True)

        if savepath.exists():
            print(f"Skipping {base_sched_name}/{target_sched_name}")
            continue
        print(f"Running {base_sched_name}/{target_sched_name}")

        initial_network = get_networks(1, cloud)[0]
        initial_task_graph = get_smallest_task_graph(recipe_name)

        sa = SimulatedAnnealing(
            task_graph=initial_task_graph,
            network=initial_network,
            scheduler=target_scheduler,
            base_scheduler=base_scheduler,
            change_types=get_changes(
                cloud_type=cloud,
                recipe_name=recipe_name,
                granularity=10
            ),
            max_iterations=1000
        )

        sa.run()

        print(f"{base_sched_name}/{target_sched_name} Energy: {sa.iterations[-1].best_energy}")
        savepath.write_bytes(pickle.dumps(sa))
