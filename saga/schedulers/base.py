from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple, Union
import networkx as nx


@dataclass
class Task:
    """A task."""
    node: str
    name: str
    start: Optional[float]
    end: Optional[float]


class Scheduler(ABC): # pylint: disable=too-few-public-methods
    """An abstract class for a scheduler."""
    @abstractmethod
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: A dictionary mapping nodes to a list of tasks executed on the node.
        """
        raise NotImplementedError


class NetworkDataset(ABC):
    """Abstract class for network datasets."""
    def __init__(self, *networks: nx.Graph) -> None:
        """Initialize a network dataset.

        Args:
            *networks (nx.Graph): The networks.
        """
        self.networks = networks

    def __getitem__(self, index: int) -> nx.Graph:
        """Get a network.

        Args:
            index (int): The index of the network.

        Returns:
            nx.Graph: The network.
        """
        return self.networks[index]
        
    def __len__(self) -> int:
        """Get the number of networks in the dataset.

        Returns:
            int: The number of networks in the dataset.
        """
        return len(self.networks)

    # not abstract because it's not necessary to implement
    @staticmethod
    def generate(*args, **kwargs) -> "NetworkDataset":
        """Generate a dataset."""
        raise NotImplementedError

    # not abstract because it's not necessary to implement
    def upload(self, *args, **kwargs) -> None:
        """Upload a dataset."""
        raise NotImplementedError

    # not abstract because it's not necessary to implement
    @staticmethod
    def download(*args, **kwargs) -> "NetworkDataset":
        """Download a dataset."""
        raise NotImplementedError

class TaskGraphDataset(ABC):
    """Abstract class for task graph datasets."""
    def __init__(self, *task_graphs: nx.DiGraph) -> None:
        """Initialize a task graph dataset.

        Args:
            *task_graphs (nx.DiGraph): The task graphs.
        """
        self.task_graphs = task_graphs

    def __getitem__(self, index: int) -> nx.DiGraph:
        """Get a task graph.

        Args:
            index (int): The index of the task graph.

        Returns:
            nx.DiGraph: The task graph.
        """
        return self.task_graphs[index]

    def __len__(self) -> int:
        """Get the number of task graphs in the dataset.

        Returns:
            int: The number of task graphs in the dataset.
        """
        return len(self.task_graphs)

    # not abstract because it's not necessary to implement
    @staticmethod
    def generate(*args, **kwargs) -> "TaskGraphDataset":
        """Generate a dataset."""
        raise NotImplementedError

    # not abstract because it's not necessary to implement
    def upload(self, *args, **kwargs) -> None:
        """Upload a dataset."""
        raise NotImplementedError

    # not abstract because it's not necessary to implement
    @staticmethod
    def download(*args, **kwargs) -> "TaskGraphDataset":
        """Download a dataset."""
        raise NotImplementedError


class MergedNetworkDataset(NetworkDataset):
    """A merged network dataset."""
    def __init__(self, *network_datasets: NetworkDataset) -> None:
        """Initialize a merged network dataset.

        Args:
            *network_datasets (NetworkDataset): The network datasets to merge.
        """
        self.network_datasets = network_datasets
        self.lengths = [len(network_dataset) for network_dataset in self.network_datasets]
        self.length = sum(self.lengths)

    def __getitem__(self, index: int) -> nx.Graph:
        """Get a network.

        Args:
            index (int): The index of the network.

        Returns:
            nx.Graph: The network.
        """
        if index < 0:
            raise IndexError("Index must be non-negative.")
        if index >= self.length:
            raise IndexError("Index out of range.")
        for i, length in enumerate(self.lengths):
            if index < length:
                return self.network_datasets[i][index]
            index -= length
        raise IndexError("Index out of range.")

    def __len__(self) -> int:
        """Get the number of networks in the dataset.

        Returns:
            int: The number of networks in the dataset.
        """
        return self.length

class MergedTaskGraphDataset(TaskGraphDataset):
    """A merged task graph dataset."""
    def __init__(self, *task_graph_datasets: TaskGraphDataset) -> None:
        """Initialize a merged task graph dataset.

        Args:
            *task_graph_datasets (TaskGraphDataset): The task graph datasets to merge.
        """
        self.task_graph_datasets = task_graph_datasets
        self.lengths = [len(task_graph_dataset) for task_graph_dataset in self.task_graph_datasets]
        self.length = sum(self.lengths)

    def __getitem__(self, index: int) -> nx.DiGraph:
        """Get a task graph.

        Args:
            index (int): The index of the task graph.

        Returns:
            nx.DiGraph: The task graph.
        """
        if index < 0:
            raise IndexError("Index must be non-negative.")
        if index >= self.length:
            raise IndexError("Index out of range.")
        for i, length in enumerate(self.lengths):
            if index < length:
                return self.task_graph_datasets[i][index]
            index -= length
        raise IndexError("Index out of range.")

    def __len__(self) -> int:
        """Get the number of task graphs in the dataset.

        Returns:
            int: The number of task graphs in the dataset.
        """
        return self.length


class Dataset(ABC):
    """Abstract class for datasets."""
    def __init__(self, *datasets: Union[NetworkDataset, TaskGraphDataset]) -> None:
        """Initialize a dataset.

        Args:
            *datasets (Union[NetworkDataset, TaskGraphDataset]): The datasets to merge.
        """
        network_datasets = []
        task_graph_datasets = []
        for dataset in datasets:
            if isinstance(dataset, NetworkDataset):
                network_datasets.append(dataset)
            elif isinstance(dataset, TaskGraphDataset):
                task_graph_datasets.append(dataset)
            else:
                raise TypeError("Dataset must be a NetworkDataset or a TaskGraphDataset.")
        if len(network_datasets) == 0:
            raise ValueError("At least one NetworkDataset must be provided.")
        if len(task_graph_datasets) == 0:
            raise ValueError("At least one TaskGraphDataset must be provided.")
        self.network_dataset = MergedNetworkDataset(*network_datasets)
        self.task_graph_dataset = MergedTaskGraphDataset(*task_graph_datasets)

    def __getitem__(self, index: int) -> Tuple[nx.Graph, nx.DiGraph]:
        """Get a network and a task graph.

        Args:
            index (int): The index of the network and task graph.

        Returns:
            Tuple[nx.Graph, nx.DiGraph]: The network and task graph.
        """
        if index < 0:
            raise IndexError("Index must be non-negative.")
        if index >= len(self.network_dataset) * len(self.task_graph_dataset):
            raise IndexError("Index out of range.")
        network_index = index // len(self.task_graph_dataset)
        task_graph_index = index % len(self.task_graph_dataset)
        return self.network_dataset[network_index], self.task_graph_dataset[task_graph_index]

    def __len__(self) -> int:
        """Get the number of networks and task graphs in the dataset.

        Returns:
            int: The number of networks and task graphs in the dataset.
        """
        return len(self.network_dataset) * len(self.task_graph_dataset)

    # not abstract because it's not necessary to implement
    @staticmethod
    def generate(*args, **kwargs) -> "Dataset":
        """Generate a dataset."""
        raise NotImplementedError

    # not abstract because it's not necessary to implement
    def upload(self, *args, **kwargs) -> None:
        """Upload a dataset."""
        raise NotImplementedError

    # not abstract because it's not necessary to implement
    @staticmethod
    def download(*args, **kwargs) -> "Dataset":
        """Download a dataset."""
        raise NotImplementedError
