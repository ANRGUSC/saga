from functools import partial
from itertools import product
from typing import Callable, List

import numpy as np

from saga import Network, TaskGraph


def get_fog_networks(
    num: int,
    num_edges_nodes: int = 100,
    num_fog_nodes: int = 5,
    num_cloud_nodes: int = 6,
    edge_node_cpu: float = 1.0,
    fog_node_cpu: float = 8.0,
    cloud_node_cpu: float = 50.0,
    edge_fog_bw: float = 60,
    fog_cloud_bw: float = 100,
    fog_fog_bw: float = 100,
    cloud_cloud_bw: float = 1e9,
) -> List[Network]:
    """Generate a fog network with the given parameters.

    Args:
        num: Number of networks to generate.
        num_edges_nodes: Number of edge nodes in each network.
        num_fog_nodes: Number of fog nodes in each network.
        num_cloud_nodes: Number of cloud nodes in each network.
        edge_node_cpu: CPU capacity of edge nodes.
        fog_node_cpu: CPU capacity of fog nodes.
        cloud_node_cpu: CPU capacity of cloud nodes.
        edge_fog_bw: Bandwidth between edge and fog nodes (Mbps).
        fog_cloud_bw: Bandwidth between fog and cloud nodes (Mbps).
        fog_fog_bw: Bandwidth between fog nodes (Mbps).
        cloud_cloud_bw: Bandwidth between cloud nodes (Mbps).

    Returns:
        A list of networks.
    """
    networks = []
    for _ in range(num):
        edge_nodes = {f"E{i}" for i in range(num_edges_nodes)}
        fog_nodes = {f"F{i}" for i in range(num_fog_nodes)}
        cloud_nodes = {f"C{i}" for i in range(num_cloud_nodes)}

        nodes = (
            [(name, edge_node_cpu) for name in edge_nodes]
            + [(name, fog_node_cpu) for name in fog_nodes]
            + [(name, cloud_node_cpu) for name in cloud_nodes]
        )
        all_node_names = edge_nodes | fog_nodes | cloud_nodes

        edge_edge_bw = edge_fog_bw
        edge_cloud_bw = min(edge_fog_bw, fog_cloud_bw)

        edges = []
        for src, dst in product(all_node_names, all_node_names):
            if src == dst:
                edges.append((src, dst, 1e9))
                continue

            has_edge = src in edge_nodes or dst in edge_nodes
            has_fog = src in fog_nodes or dst in fog_nodes
            has_cloud = src in cloud_nodes or dst in cloud_nodes

            if has_edge and not (has_fog or has_cloud):  # edge to edge
                bw = edge_edge_bw
            elif has_edge and has_fog:  # edge to fog
                bw = edge_fog_bw
            elif has_edge and has_cloud:  # edge to cloud
                bw = edge_cloud_bw
            elif has_fog and not (has_edge or has_cloud):  # fog to fog
                bw = fog_fog_bw
            elif has_fog and has_cloud:  # fog to cloud
                bw = fog_cloud_bw
            elif has_cloud and not (has_edge or has_fog):  # cloud to cloud
                bw = cloud_cloud_bw
            else:
                bw = 0.0

            # Scale by 125 so units are KiloBytes per second
            edges.append((src, dst, bw * 125))

        networks.append(Network.create(nodes=nodes, edges=edges))

    return networks


def gaussian(min_value: float, max_value: float) -> float:
    """Sample from a Gaussian distribution."""
    value = np.random.normal(
        loc=(min_value + max_value) / 2, scale=(max_value - min_value) / 3
    )
    return max(min_value, min(max_value, value))


def get_etl_task_graphs(
    num: int,
    get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
    get_task_cost: Callable[[], float] = partial(gaussian, 10, 50),
    window_1_size: int = 10,
    window_2_size: int = 10,
) -> List[TaskGraph]:
    """Generate a task graph with the given parameters.

    Args:
        num: Number of task graphs to generate.
        get_input_size: Function that returns the input size of a task.
        get_task_cost: Function that returns the cost of a task.
        window_1_size: Size of the first window.
        window_2_size: Size of the second window.

    Returns:
        A list of task graphs.
    """
    task_graphs = []
    for _ in range(num):
        input_size = get_input_size()

        tasks = [
            ("Source", get_task_cost()),
            ("SenMLParse", get_task_cost()),
            ("RangeFilter", get_task_cost()),
            ("BloomFilter", get_task_cost()),
            ("Interpolation", get_task_cost()),
            ("Join", get_task_cost()),
            ("Annotate", get_task_cost()),
            ("CsvToSenML", get_task_cost()),
            ("AzureTableInsert", get_task_cost()),
            ("MQTTPublish", get_task_cost()),
            ("Sink", get_task_cost()),
        ]

        dependencies = [
            ("Source", "SenMLParse", input_size),
            ("SenMLParse", "RangeFilter", window_1_size * input_size),
            ("RangeFilter", "BloomFilter", window_1_size * input_size),
            ("BloomFilter", "Interpolation", window_1_size * input_size),
            ("Interpolation", "Join", window_1_size * input_size),
            ("Join", "Annotate", input_size),
            ("Annotate", "CsvToSenML", input_size),
            ("Annotate", "AzureTableInsert", window_2_size * input_size),
            ("CsvToSenML", "MQTTPublish", input_size),
            ("AzureTableInsert", "Sink", window_2_size * input_size),
            ("MQTTPublish", "Sink", input_size),  # Added to ensure single sink
        ]

        task_graphs.append(TaskGraph.create(tasks=tasks, dependencies=dependencies))

    return task_graphs


def get_stats_task_graphs(
    num: int,
    get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
    get_task_cost: Callable[[], float] = partial(gaussian, 10, 50),
    window_1_size: int = 10,
    window_2_size: int = 10,
) -> List[TaskGraph]:
    """Generate a task graph with the given parameters.

    Args:
        num: Number of task graphs to generate.
        get_input_size: Function that returns the input size of a task.
        get_task_cost: Function that returns the cost of a task.
        window_1_size: Size of the first window.
        window_2_size: Size of the second window.

    Returns:
        A list of task graphs.
    """
    task_graphs = []
    for _ in range(num):
        input_size = get_input_size()
        input_size_group_viz = (
            window_1_size * input_size + 2 * input_size
        ) * window_2_size

        tasks = [
            ("Source", get_task_cost()),
            ("SenMLParse", get_task_cost()),
            ("Average", get_task_cost()),
            ("KalmanFilter", get_task_cost()),
            ("DistinctCount", get_task_cost()),
            ("SlidingLinearReg", get_task_cost()),
            ("GroupViz", get_task_cost()),
            ("BlobUpload", get_task_cost()),
            ("Sink", get_task_cost()),
        ]

        dependencies = [
            ("Source", "SenMLParse", input_size),
            ("SenMLParse", "Average", window_1_size * input_size),
            ("SenMLParse", "KalmanFilter", input_size),
            ("SenMLParse", "DistinctCount", input_size),
            ("KalmanFilter", "SlidingLinearReg", input_size),
            ("Average", "GroupViz", window_1_size * input_size),
            ("SlidingLinearReg", "GroupViz", input_size),
            ("DistinctCount", "GroupViz", input_size),
            ("GroupViz", "BlobUpload", input_size_group_viz),
            ("BlobUpload", "Sink", input_size_group_viz),
        ]

        task_graphs.append(TaskGraph.create(tasks=tasks, dependencies=dependencies))

    return task_graphs


def get_train_task_graphs(
    num: int,
    get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
    get_task_cost: Callable[[], float] = partial(gaussian, 10, 50),
    window_1_size: int = 10,
    window_2_size: int = 10,
) -> List[TaskGraph]:
    """Generate a task graph with the given parameters.

    Args:
        num: Number of task graphs to generate.
        get_input_size: Function that returns the input size of a task.
        get_task_cost: Function that returns the cost of a task.
        window_1_size: Size of the first window.
        window_2_size: Size of the second window.

    Returns:
        A list of task graphs.
    """
    task_graphs = []
    for _ in range(num):
        input_size = get_input_size()

        tasks = [
            ("TimerSource", get_task_cost()),
            ("TableRead", get_task_cost()),
            ("Annotation", get_task_cost()),
            ("MultiVarLinearRegTrain", get_task_cost()),
            ("DecisionTreeTrain", get_task_cost()),
            ("BlobWrite", get_task_cost()),
            ("MQTTPublish", get_task_cost()),
            ("Sink", get_task_cost()),
        ]

        dependencies = [
            ("TimerSource", "TableRead", input_size),
            ("TableRead", "Annotation", input_size),
            ("TableRead", "MultiVarLinearRegTrain", input_size),
            ("Annotation", "BlobWrite", input_size),
            ("MultiVarLinearRegTrain", "BlobWrite", input_size),
            ("BlobWrite", "MQTTPublish", 2 * input_size),
            ("MQTTPublish", "Sink", 2 * input_size),
        ]

        task_graphs.append(TaskGraph.create(tasks=tasks, dependencies=dependencies))

    return task_graphs


def get_predict_task_graphs(
    num: int,
    get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
    get_task_cost: Callable[[], float] = partial(gaussian, 10, 50),
    window_1_size: int = 10,
    window_2_size: int = 10,
) -> List[TaskGraph]:
    """Generate a task graph with the given parameters.

    Args:
        num: Number of task graphs to generate.
        get_input_size: Function that returns the input size of a task.
        get_task_cost: Function that returns the cost of a task.
        window_1_size: Size of the first window.
        window_2_size: Size of the second window.

    Returns:
        A list of task graphs.
    """
    task_graphs = []
    for _ in range(num):
        input_size = get_input_size()

        # Original has two sources (Source, MQTTSubscribe), so we add a virtual source
        tasks = [
            ("VirtualSource", 1e-9),  # Virtual source to ensure single source
            ("Source", get_task_cost()),
            ("MQTTSubscribe", get_task_cost()),
            ("BlobRead", get_task_cost()),
            ("SenMLParse", get_task_cost()),
            ("DecisionTree", get_task_cost()),
            ("MultiVarLinearReg", get_task_cost()),
            ("Average", get_task_cost()),
            ("ErrorEstimate", get_task_cost()),
            ("MQTTPublish", get_task_cost()),
            ("Sink", get_task_cost()),
        ]

        dependencies = [
            ("VirtualSource", "Source", 1e-9),
            ("VirtualSource", "MQTTSubscribe", 1e-9),
            ("Source", "SenMLParse", input_size),
            ("MQTTSubscribe", "BlobRead", input_size),
            ("BlobRead", "DecisionTree", input_size),
            ("BlobRead", "MultiVarLinearReg", input_size),
            ("SenMLParse", "DecisionTree", input_size),
            ("SenMLParse", "MultiVarLinearReg", input_size),
            ("SenMLParse", "Average", window_1_size * input_size),
            ("MultiVarLinearReg", "ErrorEstimate", 2 * input_size),
            ("Average", "ErrorEstimate", 2 * input_size),
            ("ErrorEstimate", "MQTTPublish", 2 * input_size),
            ("DecisionTree", "MQTTPublish", 2 * input_size),
            ("MQTTPublish", "Sink", 2 * input_size),
        ]

        task_graphs.append(TaskGraph.create(tasks=tasks, dependencies=dependencies))

    return task_graphs


def test():
    """Test the fog network generator."""
    network = get_fog_networks(num=1)[0]
    print(f"Network has {len(network.nodes)} nodes")


if __name__ == "__main__":
    test()
