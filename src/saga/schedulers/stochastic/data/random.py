import numpy as np
from typing import Callable, Hashable, List
import networkx as nx
from saga.utils.random_variable import RandomVariable
from saga.schedulers.data.random import gen_in_trees as _gen_in_trees
from saga.schedulers.data.random import gen_out_trees as _gen_out_trees
from saga.schedulers.data.random import gen_parallel_chains as _gen_parallel_chains
from saga.schedulers.data.random import gen_random_networks as _gen_random_networks

# Task Graph Datasets
def _default_task_weight(task: Hashable) -> float: # pylint: disable=unused-argument
    """Default task weight function.

    Args:
        task (Hashable): The task.

    Returns:
        float: The task weight.
    """
    samples = np.random.normal(1, 1/3, RandomVariable.DEFAULT_NUM_SAMPLES)
    samples = np.clip(samples, 1e-9, 2)
    return RandomVariable(samples=samples)

def _default_dependency_weight(src: Hashable, dst: Hashable) -> float: # pylint: disable=unused-argument
    """Default dependency weight function.

    Args:
        src (Hashable): The source node.
        dst (Hashable): The destination node.

    Returns:
        float: The dependency weight.
    """
    samples = np.random.normal(1, 1/3, RandomVariable.DEFAULT_NUM_SAMPLES)
    samples = np.clip(samples, 1e-9, 2)
    return RandomVariable(samples=samples)

def gen_in_trees(num: int, # pylint: disable=arguments-differ
                  num_levels: int,
                  branching_factor: int,
                  get_task_weight: Callable[[Hashable], float] = None,
                  get_dependency_weight: Callable[[Hashable, Hashable], float] = None) -> List[nx.DiGraph]:
    """Generate a dataset of in-trees.

    Args:
        num: Number of graphs to generate.
        num_levels: Number of levels in the tree.
        branching_factor: Number of parents per node.
        get_task_weight: A function that returns the weight of a task.
        get_dependency_weight: A function that returns the weight of a dependency.

    Returns:
        A list of in-trees.
    """
    trees = _gen_in_trees(
        num=num,
        num_levels=num_levels,
        branching_factor=branching_factor,
        get_task_weight=get_task_weight or _default_task_weight,
        get_dependency_weight=get_dependency_weight or _default_dependency_weight,
    )
    return trees

def gen_random_networks(num: int, # pylint: disable=arguments-differ
                        num_nodes: int,
                        get_node_weight: Callable[[Hashable], float] = None,
                        get_edge_weight: Callable[[Hashable, Hashable], float] = None) -> List[nx.Graph]:
    """Generate a dataset of random networks.

    Args:
        num: Number of graphs to generate.
        num_nodes: Number of nodes in the network.
        get_node_weight: A function that returns the weight of a node.
        get_edge_weight: A function that returns the weight of an edge.

    Returns:
        A list of random networks.
    """
    networks = _gen_random_networks(
        num=num,
        num_nodes=num_nodes,
        get_node_weight=get_node_weight or _default_task_weight,
        get_edge_weight=get_edge_weight or _default_dependency_weight,
    )
    return networks

def gen_out_trees(num: int, # pylint: disable=arguments-differ
                  num_levels: int,
                  branching_factor: int,
                  get_task_weight: Callable[[Hashable], float] = None,
                  get_dependency_weight: Callable[[Hashable, Hashable], float] = None) -> List[nx.DiGraph]:
    """Generate a dataset of in-trees.

    Args:
        num: Number of graphs to generate.
        num_levels: Number of levels in the tree.
        branching_factor: Number of parents per node.
        get_task_weight: A function that returns the weight of a task.
        get_dependency_weight: A function that returns the weight of a dependency.

    Returns:
        A list of out-trees.
    """
    trees = _gen_out_trees(
        num=num,
        num_levels=num_levels,
        branching_factor=branching_factor,
        get_task_weight=get_task_weight or _default_task_weight,
        get_dependency_weight=get_dependency_weight or _default_dependency_weight,
    )
    return trees

def gen_parallel_chains(num: int,
                        num_chains: int,
                        chain_length: int,
                        get_task_weight: Callable[[Hashable], float] = None,
                        get_dependency_weight: Callable[[Hashable, Hashable], float] = None) -> List[nx.DiGraph]:
    """Generate a dataset of parallel chains.

    Args:
        num: Number of graphs to generate.
        num_chains: Number of chains in the graph.
        chain_length: Length of each chain.
        get_task_weight: A function that returns the weight of a task.
        get_dependency_weight: A function that returns the weight of a dependency.

    Returns:
        A dataset of parallel chains.
    """
    chains = _gen_parallel_chains(
        num=num,
        num_chains=num_chains,
        chain_length=chain_length,
        get_task_weight=get_task_weight or _default_task_weight,
        get_dependency_weight=get_dependency_weight or _default_dependency_weight,
    )
    return chains
                  