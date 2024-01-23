from functools import lru_cache
import logging
import pathlib
import random
import re
import zipfile
from typing import Callable, Dict, Hashable, List, Optional

import networkx as nx
import xml.etree.ElementTree as ET
import requests
from io import BytesIO

from saga.data import Dataset


datapath = pathlib.Path.home() / ".saga/listvscluster"

def get_network(num_nodes: int) -> nx.Graph:
    # complete homogeneous graph
    graph = nx.complete_graph(num_nodes)
    for node in graph.nodes:
        graph.nodes[node]["weight"] = 1
        graph.add_edge(node, node, weight=1e9) # self loop with infinite comm strength

    for u, v in graph.edges:
        if u == v:
            continue
        graph.edges[u, v]["weight"] = 1

    return graph


def pull_data(repull: bool = False):
    url = "https://figshare.com/ndownloader/files/38918120"

    if datapath.exists() and not repull:
        logging.debug("Data already exists at %s", datapath)
        return

    # pull data to tempfile and unzip into home/.saga/listvscluster
    logging.debug("Pulling data from %s", url)
    res = requests.get(url)
    buffer = BytesIO()
    buffer.write(res.content)
    buffer.seek(0)
    logging.debug("Unzipping data to %s", datapath)
    with zipfile.ZipFile(buffer, 'r') as zf:
        zf.extractall(str(datapath))

def xml_to_digraph(xml_string):
    # Parse the XML string
    root = ET.fromstring(xml_string)

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for node in root.findall('.//node'):
        node_id = node.get('id')
        weight = int(node.find('.//attr[@name="Weight"]/int').text)
        G.add_node(node_id, weight=weight)

    # Add edges to the graph
    for edge in root.findall('.//edge'):
        from_node = edge.get('from')
        to_node = edge.get('to')
        weight = int(edge.find('.//attr[@name="Weight"]/int').text)
        G.add_edge(from_node, to_node, weight=weight)

    return G

def load_task_graphs(category: str) -> List[nx.DiGraph]:
    pull_data()
    logging.debug("Loading data from %s", datapath)
    graphs = []
    for path in datapath.glob(f"listSchedVSclusterSched/{category}/**/*.gxl"):
        logging.debug("Loading %s", path)
        graph = xml_to_digraph(path.read_text())
        graphs.append(graph)

    return graphs

def get_categories() -> List[str]:
    pull_data()
    logging.debug("Loading data from %s", datapath)
    categories = [path.stem for path in datapath.glob("listSchedVSclusterSched/*")]
    return categories

def load_all_task_graphs() -> Dict[str, List[nx.DiGraph]]:
    logging.debug("Loading data from %s", datapath)
    data = {}
    for category in get_categories():
        graphs = load_task_graphs(category)
        if graphs:
            data[category] = graphs

    return data

def sanitize_name(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower())
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = text.replace(" ", "_")
    return text

class LVCDataset(Dataset):
    def __init__(self, category: str):
        super().__init__(name=sanitize_name(category))
        pull_data()
        if category not in get_categories():
            raise ValueError(f"Category {category} not found")
        
        self._category = category
        self._paths = datapath.glob(f"listSchedVSclusterSched/{category}/**/*.gxl")
        
        # sort paths by path length (shortest first) and then by name (alphabetical)
        self._paths = sorted(self._paths, key=lambda path: (len(path.parts), str(path)))
        



if __name__ == "__main__":
    logging.basicConfig(level=logging.debug)
    data = load_all_task_graphs()
    print(f"Loaded: {data.keys()}")
