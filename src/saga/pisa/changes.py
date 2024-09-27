import abc
import pathlib
import random
from dataclasses import dataclass
from typing import Optional
import numpy as np
from saga.utils.random_variable import RandomVariable
from scipy.stats import norm

import networkx as nx

class Change:
    @classmethod
    def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> Optional['Change']:
        raise NotImplementedError

    @classmethod
    def apply_random(self, network: nx.Graph, task_graph: nx.DiGraph) -> 'Change':
        change: Optional[Change] = self.random(network, task_graph)
        if change is not None:
            change.apply(network, task_graph)
        return change

    @abc.abstractmethod
    def apply(self, network: nx.Graph, task_graph: nx.DiGraph) -> None:
        raise NotImplementedError

# Delete a dependency
@dataclass
class TaskGraphDeleteDependency(Change):
    task: str
    dependency: str

    @classmethod
    def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> 'Change':
        """Randomly select a dependency to delete"""
        if len(task_graph.edges) <= 0:
            return None
        task, dependency = random.choice(list(task_graph.edges))
        change = TaskGraphDeleteDependency(task, dependency)
        return change

    def apply(self, network: nx.Graph, task_graph: nx.DiGraph) -> None:
        task_graph.remove_edge(self.task, self.dependency)

def get_gaussian_random_variable() -> RandomVariable:
    """Returns a random variable with a Gaussian distribution."""
    std = np.random.uniform(0.01, 0.1)
    mean = np.random.uniform(0.5 - 4 * std, 0.5 + 4 * std)
    x = np.linspace(mean - 4 * std, mean + 4 * std, 1000)
    pdf = norm.pdf(x, loc=mean, scale=std)
    return RandomVariable.from_pdf(x, pdf)

@dataclass
class GaussianTaskGraphDeleteDependency(TaskGraphDeleteDependency):
    pass # Same as TaskGraphDeleteDependency

# Add a dependency
@dataclass
class TaskGraphAddDependency(Change):
    task: str
    dependency: str

    @classmethod
    def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> 'Change':
        """Randomly select a dependency to add"""
        # get random permutation of nodes
        nodes = list(task_graph.nodes)
        random.shuffle(nodes)

        for node in nodes:
            ancestors = nx.ancestors(task_graph, node)
            children = task_graph.successors(node)
            non_ancestors = set(nodes) - ancestors - set(children) - {node}
            if len(non_ancestors) > 0:
                dependency = random.choice(list(non_ancestors))
                change = cls(node, dependency)
                return change
        return None

    def apply(self, network: nx.Graph, task_graph: nx.DiGraph) -> None:
        weight = random.uniform(0, 1)
        task_graph.add_edge(self.task, self.dependency, weight=weight)

@dataclass
class GaussianTaskGraphAddDependency(TaskGraphAddDependency):
    task: str
    dependency: str

    # random() is inherited from TaskGraphAddDependency
    
    def apply(self, network: nx.Graph, task_graph: nx.DiGraph) -> None:
        new_weight = get_gaussian_random_variable()
        task_graph.add_edge(self.task, self.dependency, weight=new_weight)

        print(f"New edge: {self.task} -> {self.dependency} with weight {new_weight}")

# Change a dependency weight
@dataclass
class TaskGraphChangeDependencyWeight(Change):
    task: str
    dependency: str
    weight: float

    @classmethod
    def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> 'Change':
        """Randomly select a dependency to change"""
        if len(task_graph.edges) <= 0:
            return None
        task, dependency = random.choice(list(task_graph.edges))
        d_weight = random.uniform(-0.1, 0.1)
        weight = task_graph.edges[task, dependency]['weight'] + d_weight
        weight = max(1e-9, min(1, weight))
        change = TaskGraphChangeDependencyWeight(task, dependency, weight)
        return change

    def apply(self, network: nx.Graph, task_graph: nx.DiGraph) -> None:
        task_graph.edges[self.task, self.dependency]['weight'] = self.weight

@dataclass
class GaussianTaskGraphChangeDependencyWeight(TaskGraphChangeDependencyWeight):
    task: str
    dependency: str
    weight: RandomVariable

    @classmethod
    def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> 'Change':
        """Randomly select a dependency to change"""
        if len(task_graph.edges) <= 0:
            return None
        task, dependency = random.choice(list(task_graph.edges))
        new_weight = get_gaussian_random_variable()
        change = GaussianTaskGraphChangeDependencyWeight(task, dependency, new_weight)
        return change
    
    # apply() is the same as TaskGraphChangeDependencyWeight

# Change a task weight
@dataclass
class TaskGraphChangeTaskWeight(Change):
    task: str
    weight: float

    @classmethod
    def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> 'Change':
        """Randomly select a task to change"""
        if len(task_graph.nodes) <= 0:
            return None
        task = random.choice(list(task_graph.nodes))
        d_weight = random.uniform(-0.1, 0.1)
        weight = task_graph.nodes[task]['weight'] + d_weight
        weight = max(1e-9, min(1, weight))
        change = TaskGraphChangeTaskWeight(task, weight)
        return change

    def apply(self, network: nx.Graph, task_graph: nx.DiGraph) -> None:
        task_graph.nodes[self.task]['weight'] = self.weight

@dataclass
class GaussianTaskGraphChangeTaskWeight(TaskGraphChangeTaskWeight):
    task: str
    weight: RandomVariable

    @classmethod
    def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> 'Change':
        """Randomly select a task to change"""
        if len(task_graph.nodes) <= 0:
            return None
        task = random.choice(list(task_graph.nodes))
        new_weight = get_gaussian_random_variable()
        change = GaussianTaskGraphChangeTaskWeight(task, new_weight)
        return change
    
    # apply() is the same as TaskGraphChangeTaskWeight

# Change a network edge weight
@dataclass
class NetworkChangeEdgeWeight(Change):
    node1: str
    node2: str
    weight: float

    @classmethod
    def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> 'Change':
        """Randomly select an edge to change"""
        if len(network.edges) <= 0:
            return None
        # choose two unique nodes
        node1, node2 = random.sample(list(network.nodes), 2)
        d_weight = random.uniform(-0.1, 0.1)
        weight = network.edges[node1, node2]['weight'] + d_weight
        weight = max(1e-9, min(1, weight))
        change = NetworkChangeEdgeWeight(node1, node2, weight)
        return change

    def apply(self, network: nx.Graph, task_graph: nx.DiGraph) -> None:
        network.edges[self.node1, self.node2]['weight'] = self.weight

@dataclass
class GaussianNetworkChangeEdgeWeight(NetworkChangeEdgeWeight):
    node1: str
    node2: str
    weight: RandomVariable

    @classmethod
    def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> 'Change':
        """Randomly select an edge to change"""
        if len(network.edges) <= 0:
            return None
        # choose two unique nodes
        node1, node2 = random.sample(list(network.nodes), 2)
        new_weight = get_gaussian_random_variable()
        change = GaussianNetworkChangeEdgeWeight(node1, node2, new_weight)
        return change
    
    # apply() is the same as NetworkChangeEdgeWeight

# Change a network node weight
@dataclass
class NetworkChangeNodeWeight(Change):
    node: str
    weight: float

    @classmethod
    def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> 'Change':
        """Randomly select a node to change"""
        if len(network.nodes) <= 0:
            return None
        node = random.choice(list(network.nodes))
        d_weight = random.uniform(-0.1, 0.1)
        weight = network.nodes[node]['weight'] + d_weight
        weight = max(1e-9, min(1, weight))
        change = NetworkChangeNodeWeight(node, weight)
        return change

    def apply(self, network: nx.Graph, task_graph: nx.DiGraph) -> None:
        network.nodes[self.node]['weight'] = self.weight

@dataclass
class GaussianNetworkChangeNodeWeight(NetworkChangeNodeWeight):
    node: str
    weight: RandomVariable

    @classmethod
    def random(cls, network: nx.Graph, task_graph: nx.DiGraph) -> 'Change':
        """Randomly select a node to change"""
        if len(network.nodes) <= 0:
            return None
        node = random.choice(list(network.nodes))
        new_weight = get_gaussian_random_variable()
        change = GaussianNetworkChangeNodeWeight(node, new_weight)
        return change
    
    # apply() is the same as NetworkChangeNodeWeight