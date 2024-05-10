from copy import deepcopy
import json
import pathlib
from typing import Dict, List
from saga.data import serialize_graph, deserialize_graph
from saga.experiment.benchmarking.parametric.components import UpwardRanking, GreedyInsert, ParametricScheduler
from saga.utils.random_graphs import add_random_weights, get_branching_dag, get_network
import networkx as nx
import os

import numpy as np
import pygame
import random

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from saga.experiment import datadir, resultsdir, outputdir

def prepare_dataset(levels: int = 3,
                    branching_factor: int = 2,
                    num_nodes: int = 10,
                    num_problems: int = 100,
                    savepath: pathlib.Path = None):
    dataset = []
    for i in range(num_problems):
        print(f"Progress: {(i+1)/num_problems*100:.2f}%" + " " * 10, end="\r")
        task_graph = add_random_weights(get_branching_dag(levels=levels, branching_factor=branching_factor))
        network = add_random_weights(get_network(num_nodes=num_nodes))

        topological_sorts = list(nx.all_topological_sorts(task_graph))
        for topological_sort in topological_sorts:
            scheduler = ParametricScheduler(
                initial_priority=lambda *_: deepcopy(topological_sort),
                insert_task=GreedyInsert(
                    append_only=False,
                    compare="EFT",
                    critical_path=False
                )
            )

            schedule = scheduler.schedule(network.copy(), task_graph.copy())
            makespan = max(task.end for tasks in schedule.values() for task in tasks)

            dataset.append(
                {
                    "instance": i,
                    "task_graph": serialize_graph(task_graph),
                    "network": serialize_graph(network),
                    "topological_sort": deepcopy(topological_sort),
                    "makespan": makespan,
                }
            )

    print("Progress: 100.00%")

    savepath.parent.mkdir(parents=True, exist_ok=True)
    savepath.write_text(json.dumps(dataset, indent=4))

def load_dataset(cache: bool = True,
                 task_graph_levels: int = 3,
                 task_graph_branching_factor: int = 2,
                 network_nodes: int = 10,
                 num_instances: int = 100) -> List[Dict]:
    """Load the dataset from disk and deserialize the graphs.
    
    Args:
        cache (bool, optional): Whether to cache the dataset. Defaults to True.
        task_graph_levels (int, optional): Number of levels in the task graph. Defaults to 3.
        task_graph_branching_factor (int, optional): Branching factor of the task graph. Defaults to 2.
        network_nodes (int, optional): Number of nodes in the network. Defaults to 10.
        num_instances (int, optional): Number of instances to generate. Defaults to 100.

    Returns:
        List[Dict]: List of problem instances, each containing a task graph, network, topological sort, and makespan
            of HEFT on the problem instance with the given topological sort.
    """
    savedir = datadir / "ml" / f"l{task_graph_levels}_b{task_graph_branching_factor}_n{network_nodes}_i{num_instances}"
    dataset_path = savedir / "data.json"
    if not dataset_path.exists() or not cache:
        print("Generating dataset...")
        prepare_dataset(
            levels=task_graph_levels,
            branching_factor=task_graph_branching_factor,
            num_nodes=network_nodes,
            num_problems=num_instances,
            savepath=dataset_path
        )

    print("Loading dataset...")
    dataset = json.loads(dataset_path.read_text())
    # deserialize graphs
    for problem_instance in dataset:
        problem_instance["task_graph"] = deserialize_graph(problem_instance["task_graph"])
        problem_instance["network"] = deserialize_graph(problem_instance["network"])
    return dataset


def get_makespan(network: nx.Graph, task_graph: nx.DiGraph, total_ordering: List[int]) -> float:
    scheduler = ParametricScheduler(
        initial_priority=lambda *_: deepcopy(total_ordering),
        insert_task=GreedyInsert(
            append_only=False,
            compare="EFT",
            critical_path=False
        )
    )

    schedule = scheduler.schedule(network.copy(), task_graph.copy())
    makespan = max(task.end for tasks in schedule.values() for task in tasks)
    return makespan

class SchedulingEnv(gym.Env):
    def __init__(self,
                 task_graph_levels: int = 2,
                 task_graph_branching_factor: int = 2,
                 network_nodes: int = 2):
        super().__init__()

        self.task_graph_levels = task_graph_levels
        self.task_graph_branching_factor = task_graph_branching_factor
        self.network_nodes = network_nodes

        _example_task_graph = add_random_weights(get_branching_dag(levels=task_graph_levels, branching_factor=task_graph_branching_factor))

        # Observation space is the 
        # - task graph (with node/edge "weight" attribute)
        # - network (with node/edge "weight" attribute)
        self.observation_space = spaces.Tuple([
            # Stateless Stuff
            spaces.Box(low=0, high=1, shape=(1, len(_example_task_graph.nodes))), # task costs
            spaces.Box(low=0, high=1, shape=(len(_example_task_graph.nodes), len(_example_task_graph.nodes))), # dependency costs
            spaces.Box(low=0, high=1, shape=(1, network_nodes)), # node speeds
            spaces.Box(low=0, high=1, shape=(network_nodes, network_nodes)), # link speeds

            # Stateful Stuff
            # binary matrix of task graph edges (0 if no edge, 1 if edge)
            spaces.MultiBinary([len(_example_task_graph.nodes), len(_example_task_graph.nodes)]),
        ])

        # self.observation_space = spaces.Box(low=0, high=1, shape=(1, len(_example_task_graph.nodes)))
        print(self.observation_space)

        # Action space is adding an edge between two nodes in the task graph
        self.action_space = spaces.Tuple([
            spaces.Discrete(len(_example_task_graph.nodes)),
            spaces.Discrete(len(_example_task_graph.nodes)),
        ])

        self.reset()

    def reset(self, seed=None, options=None):
        # Set the random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Reset the environment to a new random problem instance
        self.task_graph = self._random_task_graph()
        self.network = self._random_network()
        return self._get_observation(), {}

    def _get_observation(self):
        # Return the current state of the environment
        # - task graph (with node/edge "weight" attribute)
        # - task graph (with node/edge "precedence" attribute)
        # - network (with node/edge "weight" attribute)
        task_costs = np.array([self.task_graph.nodes[node]["weight"] for node in self.task_graph.nodes], dtype=np.float32)
        dependency_costs = np.array([[self.task_graph[node1][node2]["weight"] if self.task_graph.has_edge(node1, node2) else 0 for node2 in self.task_graph.nodes] for node1 in self.task_graph.nodes], dtype=np.float32)
        node_speeds = np.array([self.network.nodes[node]["weight"] for node in self.network.nodes], dtype=np.float32)
        link_speeds = np.array([[self.network[node1][node2]["weight"] if self.network.has_edge(node1, node2) else 0 for node2 in self.network.nodes] for node1 in self.network.nodes], dtype=np.float32)

        task_costs = task_costs.reshape(1, -1)
        node_speeds = node_speeds.reshape(1, -1)

        # make self-edges in node_speeds == 0
        np.fill_diagonal(link_speeds, 0)

        task_graph_edges = np.array([[1 if self.task_graph.has_edge(node1, node2) else 0 for node2 in self.task_graph.nodes] for node1 in self.task_graph.nodes])
        obs = (
            task_costs,
            dependency_costs,
            node_speeds,
            link_speeds,
            task_graph_edges
        )

        return obs

    def step(self, action):
        # Apply the action to the task graph
        # If the action creates a cycle, keep the state the same and return a negative reward
        # If the action is invalid because the edge already exists, keep the state the same and return a negative reward
        # If the action is valid, update the state and return a positive reward

        # Get the two nodes to add a precedence edge between
        source, target = action

        # Check if the edge already exists and precedence=1
        if self.task_graph.has_edge(source, target) and self.task_graph[source][target].get("precedence", 0) == 1:
            return self._get_observation(), -1, False, False, {}
        
        # Check if the edge would create a cycle
        if nx.has_path(self.task_graph, target, source):
            return self._get_observation(), -2, False, False, {}

        # Add the edge with "precedence" attribute
        self.task_graph.add_edge(source, target, weight=0, precedence=1)

        # get number of edges that have precedence=1 attribute
        num_edges = sum([1 for _, _, data in self.task_graph.edges(data=True) if data.get("precedence", 0) == 1])
        # if num_edges == num_tasks - 1, then we have a total ordering
        if num_edges == len(self.task_graph.nodes) - 1:
            total_ordering = list(nx.topological_sort(self.task_graph))
            makespan = get_makespan(self.network, self.task_graph, total_ordering)
            return self._get_observation(), makespan, True, False, {}

        return self._get_observation(), 0, False, False, {}
    
    def render(self):
        pass

    def close(self):
        # pygame.quit()
        pass
        
    def _random_task_graph(self):
        return add_random_weights(get_branching_dag(levels=self.task_graph_levels, branching_factor=self.task_graph_branching_factor))
    
    def _random_network(self):
        return add_random_weights(get_network(num_nodes=self.network_nodes))

register(
    id="saga/Scheduling-v0",
    entry_point="saga.experiment.ml:SchedulingEnv",
    max_episode_steps=300,
)
