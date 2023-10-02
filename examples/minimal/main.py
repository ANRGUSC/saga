import networkx as nx
from saga.schedulers import HeftScheduler

def load_task_graph() -> nx.DiGraph:
    """Loads a task graph"""
    return nx.DiGraph()

def load_network() -> nx.Graph:
    """Loads a network"""
    return nx.Graph()

network = load_network()
task_graph = load_task_graph()
scheduler = HeftScheduler()
schedule = scheduler.schedule(network, task_graph)
