import random
import networkx as nx
from typing import Dict, Any
from abc import ABC, abstractmethod

try:
    from ..scheduler import Scheduler
    from ..utils.schedule import calculate_makespan
except ImportError:
    class Scheduler(ABC):
        @abstractmethod
        def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[str, Any]:
            pass
    
    def calculate_makespan(assignment: Dict[str, str], network: nx.Graph, task_graph: nx.DiGraph) -> float:
        """Basic makespan calculation for testing"""
        start_times = {}
        finish_times = {}
        node_available = {node: 0.0 for node in network.nodes()}
        
        for task in nx.topological_sort(task_graph):
            node = assignment[task]
            
            earliest_start = node_available[node]
            
            for pred in task_graph.predecessors(task):
                pred_finish = finish_times[pred]
                pred_node = assignment[pred]
                
                comm_cost = 0.0
                if pred_node != node:
                    edge_data = task_graph.edges.get((pred, task), {})
                    comm_cost = edge_data.get('communication_cost', 1.0)
                
                earliest_start = max(earliest_start, pred_finish + comm_cost)
            
            task_data = task_graph.nodes[task]
            comp_costs = task_data.get('computation_cost', {})
            comp_cost = comp_costs.get(node, 1.0)
            
            start_times[task] = earliest_start
            finish_times[task] = earliest_start + comp_cost
            node_available[node] = finish_times[task]
        
        return max(finish_times.values()) if finish_times else 0.0


class NRandomScheduler(Scheduler):
    
    def __init__(self, n_iterations: int = 100, seed: int = None):
        self.n_iterations = n_iterations
        if seed is not None:
            random.seed(seed)
    
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[str, Any]:
        network_nodes = list(network.nodes())
        tasks = list(task_graph.nodes())
        
        best_assignment = None
        best_makespan = float('inf')
        
        for _ in range(self.n_iterations):
            assignment = {task: random.choice(network_nodes) for task in tasks}
            
            makespan = calculate_makespan(assignment, network, task_graph)
            
            if makespan < best_makespan:
                best_makespan = makespan
                best_assignment = assignment
        
        return {
            'assignment': best_assignment,
            'makespan': best_makespan
        }


# Sample graphs for testing
def create_test_network():
    network = nx.Graph()
    network.add_nodes_from(['cpu1', 'cpu2', 'gpu1'])
    network.add_edges_from([('cpu1', 'cpu2'), ('cpu2', 'gpu1')])
    return network


def create_test_task_graph():
    task_graph = nx.DiGraph()
    
    task_graph.add_node('A', computation_cost={'cpu1': 2.0, 'cpu2': 1.5, 'gpu1': 1.0})
    task_graph.add_node('B', computation_cost={'cpu1': 3.0, 'cpu2': 2.0, 'gpu1': 1.2})
    task_graph.add_node('C', computation_cost={'cpu1': 2.5, 'cpu2': 1.8, 'gpu1': 1.1})
    
    task_graph.add_edge('A', 'B', communication_cost=0.5)
    task_graph.add_edge('A', 'C', communication_cost=0.8)
    
    return task_graph

if __name__ == "__main__":
    print("Testing N-Random Scheduler...")
    
    network = create_test_network()
    task_graph = create_test_task_graph()
    
    print(f"Network nodes: {list(network.nodes())}")
    print(f"Task nodes: {list(task_graph.nodes())}")
    print(f"Task edges: {list(task_graph.edges())}")
    
    scheduler = NRandomScheduler(n_iterations=20, seed=42)
    schedule = scheduler.schedule(network, task_graph)
    
    print("\nResults:")
    print(f"Makespan: {schedule['makespan']:.2f}")
    print(f"Assignment: {schedule['assignment']}")
    
    print("\nTesting multiple runs:")
    for i in range(3):
        sched = NRandomScheduler(n_iterations=10, seed=i)
        result = sched.schedule(network, task_graph)
        print(f"Run {i+1}: Makespan = {result['makespan']:.2f}")