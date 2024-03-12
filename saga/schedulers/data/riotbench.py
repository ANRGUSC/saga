from functools import partial
from itertools import product
from typing import Callable, List
import networkx as nx
import numpy as np

def get_fog_networks(num: int,
                     num_edges_nodes: int = 100,
                     num_fog_nodes: int = 5,
                     num_cloud_nodes: int = 6,
                     edge_node_cpu: float = 1.0,
                     fog_node_cpu: float = 8.0,
                     cloud_node_cpu: float = 50.0,
                     edge_fog_bw: float = 60,
                     fog_cloud_bw: float = 100,
                     fog_fog_bw: float = 100,
                     cloud_cloud_bw: float = 1e9) -> List[nx.DiGraph]:
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
        network = nx.Graph()
        edge_nodes = {f"E{i}" for i in range(num_edges_nodes)}
        fog_nodes = {f"F{i}" for i in range(num_fog_nodes)}
        cloud_nodes = {f"C{i}" for i in range(num_cloud_nodes)}
        network.add_nodes_from(edge_nodes, weight=edge_node_cpu)
        network.add_nodes_from(fog_nodes, weight=fog_node_cpu)
        network.add_nodes_from(cloud_nodes, weight=cloud_node_cpu)

        edge_edge_bw = edge_fog_bw
        edge_cloud = min(edge_fog_bw, fog_cloud_bw)
        for (src, dst) in product(network.nodes, network.nodes):
            if src == dst:
                network.add_edge(src, dst, weight=1e9)
            has_edge = any(n in edge_nodes for n in (src, dst))
            has_fog = any(n in fog_nodes for n in (src, dst))
            has_cloud = any(n in cloud_nodes for n in (src, dst))
            if has_edge and not (has_fog or has_cloud): # edge to edge
                network.add_edge(src, dst, weight=edge_edge_bw)
            elif has_edge and has_fog: # edge to fog
                network.add_edge(src, dst, weight=edge_fog_bw)
            elif has_edge and has_cloud: # edge to cloud
                network.add_edge(src, dst, weight=edge_cloud)
            elif has_fog and not (has_edge or has_cloud): # fog to fog
                network.add_edge(src, dst, weight=fog_fog_bw)
            elif has_fog and has_cloud: # fog to cloud
                network.add_edge(src, dst, weight=fog_cloud_bw)
            elif has_cloud and not (has_edge or has_fog): # cloud to cloud
                network.add_edge(src, dst, weight=cloud_cloud_bw)
        networks.append(network)

        # Scale all edge weights by 125 so units are KiloBytes per second
        # to match the units of the task graph.
        for (src, dst) in network.edges:
            network.edges[src, dst]["weight"] *= 125

        # Make self-loops infinite
        for node in network.nodes:
            network.edges[node, node]["weight"] = 1e9
    return networks

def gaussian(min_value: float, max_value: float) -> float:
    """Sample from a Gaussian distribution."""
    value = np.random.normal(
        loc=(min_value + max_value) / 2,
        scale=(max_value - min_value) / 3
    )
    return max(min_value, min(max_value, value))

def get_etl_task_graphs(num: int,
                        get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
                        get_task_cost: Callable[[], float] = partial(gaussian, 10, 50),
                        window_1_size: int = 10,
                        window_2_size: int = 10) -> List[nx.DiGraph]:
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
        task_graph = nx.DiGraph()
        task_graph.add_node("Source", weight=get_task_cost())
        task_graph.add_node("SenMLParse", weight=get_task_cost())
        task_graph.add_node("RangeFilter", weight=get_task_cost())
        task_graph.add_node("BloomFilter", weight=get_task_cost())
        task_graph.add_node("Interpolation", weight=get_task_cost())
        task_graph.add_node("Join", weight=get_task_cost())
        task_graph.add_node("Annotate", weight=get_task_cost())
        task_graph.add_node("CsvToSenML", weight=get_task_cost())
        task_graph.add_node("AzureTableInsert", weight=get_task_cost())
        task_graph.add_node("MQTTPublish", weight=get_task_cost())
        task_graph.add_node("Sink", weight=get_task_cost())

        input_size = get_input_size()
        task_graph.add_edge("Source", "SenMLParse", weight=input_size)
        task_graph.add_edge("SenMLParse", "RangeFilter", weight=window_1_size*input_size)
        task_graph.add_edge("RangeFilter", "BloomFilter", weight=window_1_size*input_size)
        task_graph.add_edge("BloomFilter", "Interpolation", weight=window_1_size*input_size)
        task_graph.add_edge("Interpolation", "Join", weight=window_1_size*input_size)
        task_graph.add_edge("Join", "Annotate", weight=input_size)
        task_graph.add_edge("Annotate", "CsvToSenML", weight=input_size)
        task_graph.add_edge("Annotate", "AzureTableInsert", weight=window_2_size*input_size)
        task_graph.add_edge("CsvToSenML", "MQTTPublish", weight=input_size)
        task_graph.add_edge("AzureTableInsert", "Sink", weight=window_2_size*input_size)

        task_graphs.append(task_graph)

    return task_graphs

def get_stats_task_graphs(num: int,
                          get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
                          get_task_cost: Callable[[], float] = partial(gaussian, 10, 50),
                          window_1_size: int = 10,
                          window_2_size: int = 10) -> List[nx.DiGraph]:
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
        task_graph = nx.DiGraph()
        task_graph.add_node("Source", weight=get_task_cost())
        task_graph.add_node("SenMLParse", weight=get_task_cost())
        task_graph.add_node("Average", weight=get_task_cost())
        task_graph.add_node("KalmanFilter", weight=get_task_cost())
        task_graph.add_node("DistinctCount", weight=get_task_cost())
        task_graph.add_node("SlidingLinearReg", weight=get_task_cost())
        task_graph.add_node("GroupViz", weight=get_task_cost())
        task_graph.add_node("BlobUpload", weight=get_task_cost())
        task_graph.add_node("Sink", weight=get_task_cost())

        input_size = get_input_size()
        task_graph.add_edge("Source", "SenMLParse", weight=input_size)
        task_graph.add_edge("SenMLParse", "Average", weight=window_1_size*input_size)
        task_graph.add_edge("SenMLParse", "KalmanFilter", weight=input_size)
        task_graph.add_edge("SenMLParse", "DistinctCount", weight=input_size)
        task_graph.add_edge("KalmanFilter", "SlidingLinearReg", weight=input_size)
        task_graph.add_edge("Average", "GroupViz", weight=window_1_size*input_size)
        task_graph.add_edge("SlidingLinearReg", "GroupViz", weight=input_size)
        task_graph.add_edge("DistinctCount", "GroupViz", weight=input_size)

        input_size_group_viz = (window_1_size*input_size + 2*input_size) * window_2_size
        task_graph.add_edge("GroupViz", "BlobUpload", weight=input_size_group_viz)
        task_graph.add_edge("BlobUpload", "Sink", weight=input_size_group_viz)

        task_graphs.append(task_graph)

    return task_graphs

def get_train_task_graphs(num: int,
                          get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
                          get_task_cost: Callable[[], float] = partial(gaussian, 10, 50),
                          window_1_size: int = 10,
                          window_2_size: int = 10) -> List[nx.DiGraph]:
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
        task_graph = nx.DiGraph()
        task_graph.add_node("TimerSource", weight=get_task_cost())
        task_graph.add_node("TableRead", weight=get_task_cost())
        task_graph.add_node("Annotation", weight=get_task_cost())
        task_graph.add_node("Multi-Var Linear Reg. Train", weight=get_task_cost())
        task_graph.add_node("Decision Tree Train", weight=get_task_cost())
        task_graph.add_node("Blob Write", weight=get_task_cost())
        task_graph.add_node("MQTT Publish", weight=get_task_cost())
        task_graph.add_node("Sink", weight=get_task_cost())

        input_size = get_input_size()
        task_graph.add_edge("TimerSource", "TableRead", weight=input_size)
        task_graph.add_edge("TableRead", "Annotation", weight=input_size)
        task_graph.add_edge("TableRead", "Multi-Var Linear Reg. Train", weight=input_size)
        task_graph.add_edge("Annotation", "Blob Write", weight=input_size)
        task_graph.add_edge("Multi-Var Linear Reg. Train", "Blob Write", weight=input_size)
        task_graph.add_edge("Blob Write", "MQTT Publish", weight=2*input_size)
        task_graph.add_edge("MQTT Publish", "Sink", weight=2*input_size)

        task_graphs.append(task_graph)

    return task_graphs

def get_predict_task_graphs(num: int,
                            get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
                            get_task_cost: Callable[[], float] = partial(gaussian, 10, 50),
                            window_1_size: int = 10,
                            window_2_size: int = 10) -> List[nx.DiGraph]:
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
        task_graph = nx.DiGraph()
        task_graph.add_node("Source", weight=get_task_cost())
        task_graph.add_node("MQTTSubscribe", weight=get_task_cost())
        task_graph.add_node("BlobRead", weight=get_task_cost())
        task_graph.add_node("SenMLParse", weight=get_task_cost())
        task_graph.add_node("DecisionTree", weight=get_task_cost())
        task_graph.add_node("Multi-Var Linear Reg", weight=get_task_cost())
        task_graph.add_node("Average", weight=get_task_cost())
        task_graph.add_node("ErrorEstimate", weight=get_task_cost())
        task_graph.add_node("MQTTPublish", weight=get_task_cost())
        task_graph.add_node("Sink", weight=get_task_cost())

        input_size = get_input_size()
        task_graph.add_edge("Source", "SenMLParse", weight=input_size)
        task_graph.add_edge("MQTTSubscribe", "BlobRead", weight=input_size)

        task_graph.add_edge("BlobRead", "DecisionTree", weight=input_size)
        task_graph.add_edge("BlobRead", "Multi-Var Linear Reg", weight=input_size)
        task_graph.add_edge("SenMLParse", "DecisionTree", weight=input_size)
        task_graph.add_edge("SenMLParse", "Multi-Var Linear Reg", weight=input_size)
        task_graph.add_edge("SenMLParse", "Average", weight=window_1_size*input_size)

        task_graph.add_edge("Multi-Var Linear Reg", "ErrorEstimate", weight=2*input_size)
        task_graph.add_edge("Average", "ErrorEstimate", weight=2*input_size)

        task_graph.add_edge("ErrorEstimate", "MQTTPublish", weight=2*input_size)
        task_graph.add_edge("DecisionTree", "MQTTPublish", weight=2*input_size)

        task_graph.add_edge("MQTTPublish", "Sink", weight=2*input_size)

        task_graphs.append(task_graph)

    return task_graphs



def test():
    """Test the fog network generator."""
    network = get_fog_networks(num=1)[0]
    print(network.nodes)
    print(network.edges)

if __name__ == "__main__":
    test()
