import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Hashable, Iterator, List, Optional, Tuple, Union

import networkx as nx
import pandas as pd
from joblib import Parallel, delayed

from .scheduler import Scheduler, Task
from .utils.tools import standardize_task_graph, validate_simple_schedule
from .utils.random_variable import RandomVariable

def isinstance_stochastic(network: nx.Graph, task_graph: nx.DiGraph) -> bool:
    """Check if the network and task graph are stochastic.

    Args:
        network (nx.Graph): The network.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        bool: Whether the network and task graph are stochastic.
    """
    return any(isinstance(node["weight"], RandomVariable) for node in network.nodes.values()) or \
           any(isinstance(edge["weight"], RandomVariable) for edge in network.edges.values()) or \
           any(isinstance(task["weight"], RandomVariable) for task in task_graph.nodes.values()) or \
           any(isinstance(dep["weight"], RandomVariable) for dep in task_graph.edges.values())

def get_deterministic_makespan(schedule: Dict[Hashable, List[Task]]) -> float:
    """Get the makespan of the schedule.

    Args:
        schedule (Dict[Hashable, List[Task]]): A schedule mapping nodes to a list of tasks.

    Returns:
        float: The makespan of the schedule.
    """
    return max((max(task.end for task in tasks) for tasks in schedule.values()))

def determinize_solution(network: nx.Graph,
                         task_graph: nx.DiGraph,
                         schedule: Dict[Hashable, List[Task]]) -> Tuple[nx.Graph, nx.DiGraph, Dict[Hashable, List[Task]]]:
    """Determinize the problem instance and solution.

    Args:
        network (nx.Graph): The network.
        task_graph (nx.DiGraph): The task graph.
        schedule (Dict[Hashable, List[Task]]): A schedule mapping nodes to a list of tasks.

    Returns:
        Tuple[nx.Graph, nx.DiGraph, Dict[Hashable, List[Task]]]: The determinized network, task graph, and schedule.
    """
    # determinize network
    new_network = nx.Graph()
    for node, node_data in network.nodes.items():
        weight = node_data["weight"]
        if isinstance(weight, RandomVariable):
            new_network.add_node(node, weight=max(1e-9, weight.sample()))
        else:
            new_network.add_node(node, weight=weight)

    for edge, edge_data in network.edges.items():
        weight = edge_data["weight"]
        if isinstance(weight, RandomVariable):
            new_network.add_edge(edge[0], edge[1], weight=max(1e-9, weight.sample()))
        else:
            new_network.add_edge(edge[0], edge[1], weight=weight)

    # determinize task graph
    new_task_graph = nx.DiGraph()
    for task, task_data in task_graph.nodes.items():
        weight = task_data["weight"]
        if isinstance(weight, RandomVariable):
            new_task_graph.add_node(task, weight=max(1e-9, weight.sample()))
        else:
            new_task_graph.add_node(task, weight=weight)

    for dep, dep_data in task_graph.edges.items():
        weight = dep_data["weight"]
        if isinstance(weight, RandomVariable):
            new_task_graph.add_edge(dep[0], dep[1], weight=max(1e-9, weight.sample()))
        else:
            new_task_graph.add_edge(dep[0], dep[1], weight=weight)

    task_map = {}
    for node, tasks in schedule.items():
        for task in tasks:
            task_map[task.name] = task

    task_order = sorted(task_map.values(), key=lambda task: task.start)
    new_schedule: Dict[Hashable, List[Task]] = {node: [] for node in new_network.nodes}
    new_task_map: Dict[str, Task] = {}
    for task in task_order:
        node = task.node
        # compute time to get data from dependencies
        data_available_time = max(
            [
                new_task_map[dep[0]].end + new_task_graph.edges[dep]["weight"] / new_network.edges[new_task_map[dep[0]].node, node]["weight"]
                for dep in new_task_graph.in_edges(task.name)
            ],
            default=0
        )
        start_time = data_available_time if not new_schedule[node] else max(data_available_time, new_schedule[node][-1].end)
        new_task = Task(node, task.name, start_time, start_time + new_task_graph.nodes[task.name]["weight"] / new_network.nodes[node]["weight"])
        new_schedule[node].append(new_task)
        new_task_map[task.name] = new_task


    return new_network, new_task_graph, new_schedule

def serialize_graph(graph: Union[nx.Graph, nx.DiGraph]):
    """Serialize a task graph or network.

    Args:
        graph (nx.Graph): The task graph or network.

    Returns:
        Dict: The serialized network.
    """
    link_data = nx.node_link_data(graph)
    # check if weights are random variables
    for resource in [*link_data["nodes"], *link_data["links"]]:
        if isinstance(resource["weight"], RandomVariable):
            resource["weight"] = {
                "type": "random_variable",
                "samples": resource["weight"].samples.tolist(),
            }
        else:
            resource["weight"] = {
                "type": "float",
                "value": resource["weight"],
            }

    return link_data

def deserialize_graph(graph_data: Dict) -> Union[nx.Graph, nx.DiGraph]:
    """Deserialize a task graph or network.

    Args:
        graph_data (Dict): The serialized network or task graph.

    Returns:
        Union[nx.Graph, nx.DiGraph]: The task graph or network.
    """
    # check if weights are random variables
    for resource in [*graph_data["nodes"], *graph_data["links"]]:
        if isinstance(resource["weight"], dict):
            if resource["weight"]["type"] == "random_variable":
                resource["weight"] = RandomVariable(samples=resource["weight"]["samples"])
            else:
                resource["weight"] = resource["weight"]["value"]

    return nx.node_link_graph(graph_data)

@dataclass
class Evaluation:
    """Evaluation of a scheduler on a dataset."""
    dataset: "Dataset"
    schedulers: List[Scheduler]
    makespans: List[float]

    def __getitem__(self, index: Union[int, slice]) -> Union[Tuple[nx.Graph, nx.DiGraph, float], "Evaluation"]:
        if isinstance(index, int):
            if index < 0 or index >= len(self):
                raise IndexError(f"Index {index} out of range")
            network, task_graph = self.dataset[index]
            makespan = self.makespans[index]
            return network, task_graph, makespan
        elif isinstance(index, slice):
            return Evaluation(self.dataset, self.schedulers, self.makespans[index])
        else:
            raise TypeError(f"Index {index} must be an int or slice")


    def __len__(self) -> int:
        """Get the number of networks and task graphs in the dataset.

        Returns:
            int: The number of networks and task graphs in the dataset.
        """
        return len(self.dataset)

    def __iter__(self) -> "Iterator[Tuple[nx.Graph, nx.DiGraph, float]]":
        """Get an iterator over the networks, task graphs, and makespans.

        Returns:
            Iterator[Tuple[nx.Graph, nx.DiGraph, float]]: An iterator over the networks, task graphs, and makespans.
        """
        for index in range(len(self)):
            yield self[index]

    # stats
    def mean_makespan(self) -> float:
        """Get the average makespan of the scheduler on the dataset.

        Returns:
            float: The average makespan of the scheduler on the dataset.
        """
        return sum(self.makespans) / len(self)

    def min_makespan(self) -> float:
        """Get the minimum makespan of the scheduler on the dataset.

        Returns:
            float: The minimum makespan of the scheduler on the dataset.
        """
        return min(self.makespans)

    def max_makespan(self) -> float:
        """Get the maximum makespan of the scheduler on the dataset.

        Returns:
            float: The maximum makespan of the scheduler on the dataset.
        """
        return max(self.makespans)

    def median_makespan(self) -> float:
        """Get the median makespan of the scheduler on the dataset.

        Returns:
            float: The median makespan of the scheduler on the dataset.
        """
        return sorted(self.makespans)[len(self) // 2]

    def std_makespan(self) -> float:
        """Get the standard deviation of the makespans of the scheduler on the dataset.

        Returns:
            float: The standard deviation of the makespans of the scheduler on the dataset.
        """
        mean_makespan = self.mean_makespan()
        return (sum((makespan - mean_makespan)**2 for makespan in self.makespans) / len(self))**0.5

    def variance_makespan(self) -> float:
        """Get the variance of the makespans of the scheduler on the dataset.

        Returns:
            float: The variance of the makespans of the scheduler on the dataset.
        """
        return self.std_makespan()**2

    def stats(self) -> Dict[str, float]:
        """Get the statistics of the makespans of the scheduler on the dataset.

        Returns:
            Dict[str, float]: The statistics of the makespans of the scheduler on the dataset.
        """
        return {
            "mean_makespan": self.mean_makespan(),
            "min_makespan": self.min_makespan(),
            "max_makespan": self.max_makespan(),
            "median_makespan": self.median_makespan(),
            "std_makespan": self.std_makespan(),
            "variance_makespan": self.variance_makespan(),
        }

class Comparison:
    """Comparison of schedulers on a dataset."""
    def __init__(self, evaluations: Dict[Scheduler, Evaluation]) -> None:
        """Comparison of schedulers on a dataset.

        Args:
            evaluations (Dict[Scheduler, Evaluation]): A dictionary mapping schedulers to their evaluations.
        """
        self.schedulers = {scheduler.__name__: scheduler for scheduler in evaluations.keys()}
        self.evaluations = {
            scheduler.__name__: evaluation
            for scheduler, evaluation in evaluations.items()
        }
        # check that all evaluations are the same length
        if len(set(len(evaluation) for evaluation in self.evaluations.values())) != 1:
            raise ValueError("Evaluations must be the same length")
        
    def __getitem__(self, index: Union[int, slice]) -> Union[Tuple[nx.Graph, nx.DiGraph, Dict[str, float]], "Comparison"]:
        if isinstance(index, int):
            if index < 0 or index >= len(self):
                raise IndexError(f"Index {index} out of range")
            makespans = {}
            network, task_graph = None, None
            for scheduler_name, evaluation in self.evaluations.items():
                network, task_graph, makespan = evaluation[index]
                makespans[scheduler_name] = makespan
            return network, task_graph, makespans
        elif isinstance(index, slice):
            return Comparison({
                scheduler_name: evaluation[index]
                for scheduler_name, evaluation in self.evaluations.items()
            })
        else:
            raise TypeError(f"Index {index} must be an int or slice")

    def __len__(self) -> int:
        """Get the number of schedulers in the comparison.

        Returns:
            int: The number of schedulers in the comparison.
        """
        evaluation = next(iter(self.evaluations.values()))
        return len(evaluation)

    def makespan_ratio(self, index: int) -> Tuple[nx.Graph, nx.DiGraph, Dict[str, float]]:
        """ Get the makespan ratio of the schedulers on the dataset.

        Args:
            index (int): The index of the network and task graph.

        Returns:
            Tuple[nx.Graph, nx.DiGraph, Dict[str, float]]: The network, task graph, and makespan ratios.
                The makespan ratios are a dictionary mapping scheduler names to their makespan ratios.
        """
        ratios = {}
        network, task_graph, makespans = self[index]
        min_makespan = min(makespans.values())
        ratios = {
            scheduler_name: makespan / min_makespan
            for scheduler_name, makespan in makespans.items()
        }
        return network, task_graph, ratios

    def all_makepsan_ratios(self) -> Dict[str, float]:
        """Get the makespan ratio of the schedulers on the dataset.

        Returns:
            Dict[str, float]: The makespan ratio of the schedulers on the dataset.
        """
        ratios: Dict[str, List[float]] = {}
        for index in range(len(self)):
            _, _, makespan_ratios = self.makespan_ratio(index)
            for scheduler_name, makespan_ratio in makespan_ratios.items():
                ratios.setdefault(scheduler_name, []).append(makespan_ratio)

        return {
            scheduler_name: makespan_ratios
            for scheduler_name, makespan_ratios in ratios.items()
        }

    def mean_makepsan_ratio(self) -> Dict[str, float]:
        """Get the average makespan ratio of the schedulers on the dataset.

        Returns:
            Dict[str, float]: The average makespan ratio of the schedulers on the dataset.
        """
        ratios = self.all_makepsan_ratios()
        return {
            scheduler_name: sum(makespan_ratios) / len(makespan_ratios)
            for scheduler_name, makespan_ratios in ratios.items()
        }

    def min_makepsan_ratio(self) -> Dict[str, float]:
        """Get the minimum makespan ratio of the schedulers on the dataset.

        Returns:
            Dict[str, float]: The minimum makespan ratio of the schedulers on the dataset.
        """
        ratios = self.all_makepsan_ratios()
        return {
            scheduler_name: min(makespan_ratios)
            for scheduler_name, makespan_ratios in ratios.items()
        }

    def max_makepsan_ratio(self) -> Dict[str, float]:
        """Get the maximum makespan ratio of the schedulers on the dataset.

        Returns:
            Dict[str, float]: The maximum makespan ratio of the schedulers on the dataset.
        """
        ratios = self.all_makepsan_ratios()
        return {
            scheduler_name: max(makespan_ratios)
            for scheduler_name, makespan_ratios in ratios.items()
        }

    def median_makepsan_ratio(self) -> Dict[str, float]:
        ratios = self.all_makepsan_ratios()
        return {
            scheduler_name: sorted(makespan_ratios)[len(makespan_ratios) // 2]
            for scheduler_name, makespan_ratios in ratios.items()
        }

    def std_makepsan_ratio(self) -> Dict[str, float]:
        """Get the standard deviation of the makespan ratios of the schedulers on the dataset.

        Returns:
            Dict[str, float]: The standard deviation of the makespan ratios of the schedulers on the dataset.
        """
        ratios = self.all_makepsan_ratios()
        mean_ratios = self.mean_makepsan_ratio()
        return {
            scheduler_name: (sum((makespan_ratio - mean_ratios[scheduler_name])**2 for makespan_ratio in makespan_ratios) / len(makespan_ratios))**0.5
            for scheduler_name, makespan_ratios in ratios.items()
        }

    def variance_makepsan_ratio(self) -> Dict[str, float]:
        """Get the variance of the makespan ratios of the schedulers on the dataset.

        Returns:
            Dict[str, float]: The variance of the makespan ratios of the schedulers on the dataset.
        """
        return {
            scheduler_name: std_ratio**2
            for scheduler_name, std_ratio in self.std_makepsan_ratio().items()
        }

    def stats(self) -> Dict[str, Dict[str, float]]:
        """Get the statistics of the makespan ratios of the schedulers on the dataset.

        Returns:
            Dict[str, Dict[str, float]]: The statistics of the makespan ratios of the schedulers on the dataset.
        """
        return {
            scheduler_name: {
                "mean_makespan_ratio": mean_ratio,
                "min_makespan_ratio": min_ratio,
                "max_makespan_ratio": max_ratio,
                "median_makespan_ratio": median_ratio,
                "std_makespan_ratio": std_ratio,
                "variance_makespan_ratio": variance_ratio,
            }
            for scheduler_name, mean_ratio, min_ratio, max_ratio, median_ratio, std_ratio, variance_ratio in zip(
                self.schedulers.keys(),
                self.mean_makepsan_ratio().values(),
                self.min_makepsan_ratio().values(),
                self.max_makepsan_ratio().values(),
                self.median_makepsan_ratio().values(),
                self.std_makepsan_ratio().values(),
                self.variance_makepsan_ratio().values(),
            )
        }
    
    def to_df(self) -> pd.DataFrame:
        """Convert the comparison to a dataframe.

        Returns:
            pd.DataFrame: The dataframe.
        """
        return pd.DataFrame(
            [
                [scheduler_name, makespan_ratio]
                for scheduler_name, makespan_ratios in self.all_makepsan_ratios().items()
                for makespan_ratio in makespan_ratios
            ],
            columns=["scheduler", "makespan_ratio"]
        )


class Dataset(ABC):
    """Abstract class for datasets."""
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        if name is None:
            name = self.__class__.__name__
        self.name = name

    @abstractmethod
    def __len__(self) -> int:
        """Get the number of networks and task graphs in the dataset.

        Returns:
            int: The number of networks and task graphs in the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: Union[int, slice]) -> Union[Tuple[nx.Graph, nx.DiGraph], "Dataset"]:
        """Get a network and a task graph.

        Args:
            index (Union[int, slice]): The index of the network and task graph.

        Returns:
            Tuple[nx.Graph, nx.DiGraph]: The network and task graph.
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[Tuple[nx.Graph, nx.DiGraph]]:
        """Get an iterator over the networks and task graphs.

        Returns:
            Iterator[Tuple[nx.Graph, nx.DiGraph]]: An iterator over the networks and task graphs.
        """
        for index in range(len(self)):
            yield self[index]

    def evaluate(self,
                 scheduler: Scheduler,
                 ignore_errors: bool = False) -> Evaluation:
        """Evaluate a scheduler on the dataset.

        Args:
            scheduler (Scheduler): The scheduler.

        Returns:
            Evaluation: The evaluation of the scheduler on the dataset.
        """
        makespans = []
        for i, (network, task_graph) in enumerate(self):
            print(f"Evaluating {scheduler.__name__} on {self.name} instance {i+1}/{len(self)}")
            try:
                task_graph = standardize_task_graph(task_graph)
                schedule = scheduler.schedule(network, task_graph)
                if isinstance_stochastic(network, task_graph):
                    for i in range(10): # TODO: make this a parameter
                        new_network, new_task_graph, new_schedule = determinize_solution(network, task_graph, schedule)
                        validate_simple_schedule(new_network, new_task_graph, new_schedule)
                        makespans.append(max(task.end for tasks in new_schedule.values() for task in tasks))
                else:
                    makespans.append(max(task.end for tasks in schedule.values() for task in tasks))
            except Exception as exception: # pylint: disable=broad-except
                if ignore_errors:
                    makespans.append(float("inf"))
                else:
                    raise exception
        return Evaluation(self, scheduler, makespans)

    def compare(self,
                schedulers: List[Scheduler],
                ignore_errors: bool = False,
                num_jobs: int = 1) -> Comparison:
        """Compare schedulers on the dataset.

        Args:
            schedulers (List[Scheduler]): The schedulers.
            ignore_errors (bool, optional): Whether to ignore errors. Defaults to False.
            num_jobs (int, optional): The number of jobs to run in parallel. Defaults to 1.

        Returns:
            Comparison: The comparison of the schedulers on the dataset.
        """
        if num_jobs == 1:
            evaluations = {
                scheduler: self.evaluate(scheduler, ignore_errors=ignore_errors)
                for scheduler in schedulers
            }
        else:
            evaluations = Parallel(n_jobs=num_jobs)(
                delayed(self.evaluate)(scheduler, ignore_errors=ignore_errors)
                for scheduler in schedulers
            )
            evaluations = {
                scheduler: evaluation
                for scheduler, evaluation in zip(schedulers, evaluations)
            }
        return Comparison(evaluations)

    @staticmethod
    def from_networks_and_task_graphs(networks: List[nx.Graph],
                                      task_graphs: List[nx.DiGraph],
                                      name: Optional[str] = None) -> "AllPairsDataset":
        """Create a dataset of all pairs of networks and task graphs.

        Args:
            networks (List[nx.Graph]): The networks.
            task_graphs (List[nx.DiGraph]): The task graphs.
            name (Optional[str], optional): The name of the dataset. Defaults to class name.

        Returns:
            AllPairsDataset: The dataset of all pairs of networks and task graphs.
        """
        return AllPairsDataset(networks, task_graphs, name=name)

    @staticmethod
    def from_pairs(pairs: List[Tuple[nx.Graph, nx.DiGraph]],
                   name: Optional[str] = None) -> "PairsDataset":
        """Create a dataset of all pairs of networks and task graphs.

        Args:
            pairs (List[Tuple[nx.Graph, nx.DiGraph]]): The pairs of networks and task graphs.
            name (Optional[str], optional): The name of the dataset. Defaults to class name.

        Returns:
            AllPairsDataset: The dataset of all pairs of networks and task graphs.
        """
        return PairsDataset(pairs, name=name)

    def to_json(self, *args, **kwargs) -> str:
        """Convert the dataset to a JSON object.

        Returns:
            Dict[str, Union[List[str], List[List[str]]]]: The JSON object.
        """
        raise NotImplementedError

    @classmethod
    def from_json(cls, json_str: str, *args, **kwargs) -> "Dataset":
        """Create a dataset from a JSON object.

        Args:
            json_str (str): The JSON object.

        Returns:
            Dataset: The dataset.
        """
        raise NotImplementedError


class AllPairsDataset(Dataset):
    """A dataset of all pairs of networks and task graphs."""
    def __init__(self, 
                 networks: List[nx.Graph],
                 task_graphs: List[nx.DiGraph],
                 name: Optional[str] = None) -> None:
        """A dataset of all pairs of networks and task graphs.

        Args:
            networks (List[nx.Graph]): The networks.
            task_graphs (List[nx.DiGraph]): The task graphs.
            name (Optional[str], optional): The name of the dataset. Defaults to class name.
        """
        super().__init__(name)
        self.networks = networks
        self.task_graphs = task_graphs

    def __len__(self) -> int:
        return len(self.networks) * len(self.task_graphs)

    def __getitem__(self, index: Union[int, slice]) -> Union[Tuple[nx.Graph, nx.DiGraph], "AllPairsDataset"]:
        if isinstance(index, int):
            if index < 0 or index >= len(self):
                raise IndexError(f"Index {index} out of range")
            network_index = index // len(self.task_graphs)
            task_graph_index = index % len(self.task_graphs)
            return self.networks[network_index], self.task_graphs[task_graph_index]
        elif isinstance(index, slice):
            return AllPairsDataset(self.networks[index], self.task_graphs[index], name=self.name)
        else:
            raise TypeError(f"Index {index} must be an int or slice")

    def to_json(self, *args, **kwargs) -> str:
        """Convert the dataset to a JSON object.

        Returns:
            Dict[str, Union[List[str], List[List[str]]]]: The JSON object.
        """
        data = {
            "name": self.name,
            "networks": [nx.node_link_data(network) for network in self.networks],
            "task_graphs": [nx.node_link_data(task_graph) for task_graph in self.task_graphs],
        }
        return json.dumps(data, *args, **kwargs)

    @classmethod
    def from_json(cls, json_str: str, *args, **kwargs) -> "Dataset":
        """Create a dataset from a JSON object.

        Args:
            json_str (str): The JSON object.

        Returns:
            Dataset: The dataset.
        """
        data: Dict = json.loads(json_str, *args, **kwargs)
        networks = [nx.node_link_graph(network_data) for network_data in data["networks"]]
        task_graphs = [nx.node_link_graph(task_graph_data) for task_graph_data in data["task_graphs"]]
        name = data.get("name") or None
        return cls(networks, task_graphs, name=name)


class PairsDataset(Dataset):
    """A dataset of pairs of networks and task graphs."""
    def __init__(self, pairs: List[Tuple[nx.Graph, nx.DiGraph]], name: Optional[str] = None) -> None:
        """A dataset of pairs of networks and task graphs.

        Args:
            pairs (List[Tuple[nx.Graph, nx.DiGraph]]): The pairs of networks and task graphs.
        """
        super().__init__(name)
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: Union[int, slice]) -> Union[Tuple[nx.Graph, nx.DiGraph], "PairsDataset"]:
        if isinstance(index, int):
            if index < 0 or index >= len(self):
                raise IndexError(f"Index {index} out of range")
            return self.pairs[index]
        elif isinstance(index, slice):
            return PairsDataset(self.pairs[index], name=self.name)
        else:
            raise TypeError(f"Index {index} must be an int or slice")

    def to_json(self, *args, **kwargs) -> str:
        """Convert the dataset to a JSON object.

        Returns:
            Dict[str, Union[List[str], List[List[str]]]]: The JSON object.
        """
        pairs = []
        for network, task_graph in self.pairs:
            pairs.append([serialize_graph(network), serialize_graph(task_graph)])
        data = {"name": self.name, "pairs": pairs}
        return json.dumps(data, *args, **kwargs)

    @classmethod
    def from_json(cls, json_str: str, *args, **kwargs) -> "Dataset":
        """Create a dataset from a JSON object.

        Args:
            json_str (str): The JSON object.

        Returns:
            Dataset: The dataset.
        """
        data: Dict = json.loads(json_str, *args, **kwargs)
        if isinstance(data, dict):
            name = data.get("name") or None
            pairs = [
                (
                    deserialize_graph(network_data),
                    deserialize_graph(task_graph_data),
                )
                for network_data, task_graph_data in data["pairs"]
            ]
        else: # TODO: For backwards compatibility, remove in future
            name = None
            pairs = [
                (
                    deserialize_graph(network_data),
                    deserialize_graph(task_graph_data),
                )
                for network_data, task_graph_data in data
            ]
        return cls(pairs, name=name)
