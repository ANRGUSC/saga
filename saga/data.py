from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterator, Union
import networkx as nx

from .schedulers.base import Scheduler


@dataclass
class Evaluation:
    """Evaluation of a scheduler on a dataset."""
    dataset: "Dataset"
    schedulers: List[Scheduler]
    makespans: List[float]

    def __getitem__(self, index: int) -> Tuple[nx.Graph, nx.DiGraph, float]:
        if index < 0 or index >= len(self):
            raise IndexError
        network, task_graph = self.dataset[index]
        makespan = self.makespans[index]
        return network, task_graph, makespan

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

    def __getitem__(self, index: int) -> Tuple[nx.Graph, nx.DiGraph, Dict[str, float]]:
        if index < 0 or index >= len(self):
            raise IndexError
        makespans = {}
        network, task_graph = None, None
        for scheduler_name, evaluation in self.evaluations.items():
            network, task_graph, makespan = evaluation[index]
            makespans[scheduler_name] = makespan
        return network, task_graph, makespans

    def __len__(self) -> int:
        """Get the number of schedulers in the comparison.

        Returns:
            int: The number of schedulers in the comparison.
        """
        return len(self.evaluations)

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


class Dataset(ABC):
    """Abstract class for datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Get the number of networks and task graphs in the dataset.

        Returns:
            int: The number of networks and task graphs in the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[nx.Graph, nx.DiGraph]:
        """Get a network and a task graph.

        Args:
            index (int): The index of the network and task graph.

        Returns:
            Tuple[nx.Graph, nx.DiGraph]: The network and task graph.
        """
        raise NotImplementedError

    def __iter__(self) -> "Iterator[Tuple[nx.Graph, nx.DiGraph]]":
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
        for network, task_graph in self:
            try:
                schedule = scheduler.schedule(network, task_graph)
                makespan = max(task.end for _, tasks in schedule.items() for task in tasks)
            except Exception as exception: # pylint: disable=broad-except
                if ignore_errors:
                    makespan = float("inf")
                else:
                    raise exception
            makespans.append(makespan)
        return Evaluation(self, scheduler, makespans)

    def compare(self,
                schedulers: List[Scheduler],
                ignore_errors: bool = False) -> Comparison:
        """Compare schedulers on the dataset.

        Args:
            schedulers (List[Scheduler]): The schedulers.
            ignore_errors (bool, optional): Whether to ignore errors. Defaults to False.

        Returns:
            Comparison: The comparison of the schedulers on the dataset.
        """
        evaluations = {
            scheduler: self.evaluate(scheduler, ignore_errors=ignore_errors)
            for scheduler in schedulers
        }
        return Comparison(evaluations)

    @staticmethod
    def from_networks_and_task_graphs(networks: List[nx.Graph], task_graphs: List[nx.DiGraph]) -> "AllPairsDataset":
        """Create a dataset of all pairs of networks and task graphs.

        Args:
            networks (List[nx.Graph]): The networks.
            task_graphs (List[nx.DiGraph]): The task graphs.

        Returns:
            AllPairsDataset: The dataset of all pairs of networks and task graphs.
        """
        return AllPairsDataset(networks, task_graphs)

    @staticmethod
    def from_pairs(pairs: List[Tuple[nx.Graph, nx.DiGraph]]) -> "PairsDataset":
        """Create a dataset of all pairs of networks and task graphs.

        Args:
            pairs (List[Tuple[nx.Graph, nx.DiGraph]]): The pairs of networks and task graphs.

        Returns:
            AllPairsDataset: The dataset of all pairs of networks and task graphs.
        """
        return PairsDataset(pairs)


class AllPairsDataset(Dataset):
    """A dataset of all pairs of networks and task graphs."""
    def __init__(self, networks: List[nx.Graph], task_graphs: List[nx.DiGraph]):
        """A dataset of all pairs of networks and task graphs.

        Args:
            networks (List[nx.Graph]): The networks.
            task_graphs (List[nx.DiGraph]): The task graphs.
        """
        assert len(networks) == len(task_graphs)
        self.networks = networks
        self.task_graphs = task_graphs

    def __len__(self) -> int:
        return len(self.networks) * len(self.task_graphs)

    def __getitem__(self, index: int) -> Tuple[nx.Graph, nx.DiGraph]:
        if index < 0 or index >= len(self):
            raise IndexError
        network_index = index // len(self.task_graphs)
        task_graph_index = index % len(self.task_graphs)
        return self.networks[network_index], self.task_graphs[task_graph_index]


class PairsDataset(Dataset):
    """A dataset of pairs of networks and task graphs."""
    def __init__(self, pairs: List[Tuple[nx.Graph, nx.DiGraph]]):
        """A dataset of pairs of networks and task graphs.

        Args:
            pairs (List[Tuple[nx.Graph, nx.DiGraph]]): The pairs of networks and task graphs.
        """
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[nx.Graph, nx.DiGraph]:
        if index < 0 or index >= len(self):
            raise IndexError
        return self.pairs[index]
