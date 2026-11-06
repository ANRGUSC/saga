from abc import ABC, abstractmethod
import logging
from typing import Dict, FrozenSet, Generator, Iterable, List, Optional, Set, Tuple
from pydantic import BaseModel, PrivateAttr, Field
from functools import cached_property
import math
import networkx as nx
import bisect
from itertools import product
import copy
from saga import NetworkEdge, TaskGraphNode, TaskGraph, Network, Schedule, Scheduler, NetworkNode

# Tolerance for floating point comparisons
EPS = 1e-9


class TaskGraphNodeFOW(TaskGraphNode):
    """A task graph node that is aware of the fog-of-war environment."""
    region: str = Field(..., description="The region this task belongs to.")
    name: str = super().__name__ + f"_{region}"

class TaskGraphFOW(TaskGraph):
    """A task graph that is aware of the fog-of-war environment."""

    region: str = Field(..., description="The region this task graph belongs to.")


class NetworkFOW(Network):
    """A network that is aware of the fog-of-war environment."""

    region: str = Field(..., description="The region this network belongs to.")
class NetworkNodeFOW(NetworkNode):
    """A network node that is aware of the fog-of-war environment."""
    region: str = Field(..., description="The region this network node belongs to.")
    name: str = super().__name__ + f"_{region}"


def regionize(network, task_graph, region: str) -> Tuple[NetworkFOW, TaskGraphFOW]:
    """Convert a network and task graph into their fog-of-war aware counterparts."""
    fow_network = NetworkFOW(
        name=network.name,
        nodes=network.nodes.copy(),
        edges=network.edges.copy(),
        region=region
    )
    fow_task_graph = TaskGraphFOW(
        name=task_graph.name,
        nodes=task_graph.nodes.copy(),
        edges=task_graph.edges.copy(),
        region=region
    )
    return fow_network, fow_task_graph


def connect_networks(
    net0: NetworkFOW,
    net1: NetworkFOW,
    link: float,
    bridge_0: Optional[str] = None,
    bridge_1: Optional[str] = None,
) -> NetworkFOW:
    """Connect two disconnected networks by adding edges between disconnected processors representative
    of pathing between connected processors.

    ex: A               A
     (2)|       ->.  (2)| \ (2*3)\2+3 
        B--C            B--C
         (3)             (3)
    
    To combine two full networks (0 and 1), we use the following algoritm.
    1. Connect any two disconnected processors with an edge weight, representing the path between them.
    call these processorsbridge_0 and bridge_1
    2. Create an edge between the an unconnected processor in one network, with one of the connected
    processor in the other network.
        (a) use the equation:
        for a node on 0:
        new_link = link(new_node_0, bridge_0) * link(bridge_0, bridge_1)/(link(new_node_0, bridge_0) + link(bridge_0, bridge_1))
        for a node on 1:
        new_link = link(new_node_1, bridge_1) * link(bridge_0, bridge_1)/(link(new_node_1, bridge_1) + link(bridge_0, bridge_1))
     3. Repeat step 2 for all unconnected processors in both networks.

    Every new link is a single jump a*b/(a+b) over two existing links: step 2 jumps through the
    bridge, and step 3 jumps through the bridge edge just created in step 2.

    Args:
        net0: Network 0. The returned network takes its region.
        net1: Network 1.
        link: Bandwidth of the physical link between bridge_0 and bridge_1.
        bridge_0: Node in net0 on the physical link. Defaults to the first node by name.
        bridge_1: Node in net1 on the physical link. Defaults to the first node by name.

    Returns:
        NetworkFOW: The combined network, with an edge between every pair of processors.
    """
    names_0 = {node.name for node in net0.nodes}
    names_1 = {node.name for node in net1.nodes}
    if names_0 & names_1:
        raise ValueError(f"Networks share node names: {sorted(names_0 & names_1)}")
    if link <= 0:
        raise ValueError("Link bandwidth must be positive.")

    bridge_0 = bridge_0 if bridge_0 is not None else min(names_0)
    bridge_1 = bridge_1 if bridge_1 is not None else min(names_1)
    if bridge_0 not in names_0 or bridge_1 not in names_1:
        raise ValueError(f"Bridge nodes {bridge_0}, {bridge_1} must be in net0 and net1.")

    def jump(a: float, b: float) -> float:
        return a * b / (a + b)

    bridge_edge = NetworkEdge(source = bridge_0, target = bridge_1, speed = link)
   
    # first, connect each node to the bridge node in the alternate network
    to_bridge_1 = {}
    for x in names_0:
        if x == bridge_0:
            to_bridge_1[x] = link
        else:
            to_bridge_1[x] = jump(net0.get_edge(x, bridge_0).speed, link)
    # to_bridge_1 = {x: jump(net0.get_edge(x, bridge_0).speed, link) for x in names_0}
    #links the nodes in net0 to the bridge node in net1
    #to_bridge_0 = {y: jump(link, net1.get_edge(bridge_1, y).speed) for y in names_1}

    # second, connect each node to all alternate nodes using the bridge speed
    cross_edges = [
        (x, y, to_bridge_1[x] if y == bridge_1
            else jump(to_bridge_1[x], net1.get_edge(bridge_1, y).speed))
        for x, y in product(sorted(names_0), sorted(names_1))
    ]
    #two way connections, so only need to go through bridge 1 once
    print("test")
    combined = Network.create(
        nodes=[*net0.nodes, *net1.nodes],
        edges=[*net0.edges, *net1.edges, bridge_edge, *cross_edges],
    )
    return NetworkFOW(nodes=combined.nodes, edges=combined.edges, region=net0.region)



def generate_fow_networks(regions: dict[str, NetworkFOW], mapping: Optional[dict[str, dict[str, Tuple[float, List[TaskGraphNodeFOW]]]]] = None) -> dict[str, NetworkFOW]:
    """Generate a list of fog-of-war networks for the given regions based on a base network.

    Args:
        regions (dict[str, NetworkFOW]): A dictionary of regions and their corresponding base networks.
        mapping (Optional[Dict[str, str]]): A mapping of regions to their shared bandwidth for other 
        regions. Format: {region: [(country, link, [shared nodes])]}.

    Returns:
        List[NetworkFOW]: A list of fog-of-war networks for each region.
    """
    region_set = set(regions.keys())
    combined_networks:dict[str, NetworkFOW] = {}

    for region, base_network in regions.items():
        new_network = copy.deepcopy(base_network)
        for other_region in region_set - {region}:
            #make a temporary network for the nodes the country shares with other regions
            shared_nodes = mapping[region][other_region][1]  # list of shared nodes
            link = mapping[region][other_region][0]  # bandwidth of the link
            temp_network = Network.create(
                nodes=[node for node in new_network.nodes if node.name in shared_nodes],
                edges=[edge for edge in new_network.edges if edge.source.name in shared_nodes and edge.target.name in shared_nodes],
            )
            new_network = connect_networks(new_network, temp_network,link)
        combined_networks[region] = new_network

    return combined_networks
