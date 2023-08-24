import pathlib
import pickle
from typing import Dict, Hashable, List

import networkx as nx
from saga.schedulers.base import Task
from saga.schedulers import CpopScheduler, HeftScheduler
from simulated_annealing import SimulatedAnnealing

thisdir = pathlib.Path(__file__).parent.absolute()

def load_results() -> SimulatedAnnealing:
    return pickle.loads((thisdir / "results" / "HEFT" / "CPOP.pkl").read_bytes())

def format_schedule(schedule: Dict[Hashable, List[Task]]) -> str:
    """Converts schedule to string representation.

    Args:
        schedule (Dict[Hashable, List[Task]]): [description]

    Returns:
        str: [description]
    """
    result = ""
    for node, tasks in schedule.items():
        result += f"Node {node}: "
        for task in tasks:
            result += f"[{task.name} start={task.start:.2f} end={task.end:.2f}] "
        result += "\n"
    return result

def format_task_graph(task_graph: nx.DiGraph) -> str:
    """Converts task graph to string representation.

    Args:
        task_graph (nx.DiGraph): [description]

    Returns:
        str: [description]
    """
    result = ""
    for task in nx.topological_sort(task_graph):
        result += f"[Task {task} cost={task_graph.nodes[task]['weight']:.2f}"
        for child in task_graph.successors(task):
            result += f" [dependent={child} io_size={task_graph.edges[task, child]['weight']:.2f}]"
        result += "]\n"
    return result

def format_network(network: nx.Graph) -> str:
    """Converts network to string representation.

    Args:
        network (nx.Graph): [description]

    Returns:
        str: [description]
    """
    result = ""
    # for node in sorted(network.nodes, key=lambda x: network.nodes[x]["weight"], reverse=True):
    for node in network.nodes:
        result += f"[Node {node} comp_speed={network.nodes[node]['weight']:.2f}"
        for neighbor in network.neighbors(node):
            if neighbor == node:
                continue
            result += f" [neighbor={neighbor} comm_speed={network.edges[node, neighbor]['weight']:.2f}]"
        result += "]\n"
    return result

def standardize_isntance(network: nx.Graph, task_graph: nx.DiGraph) -> None:
    # make network self-loops have weight 1e9
    for node in network.nodes:
        network.add_edge(node, node, weight=1e9)

    # add dummy src and dst nodes to task graph
    src = "__src__"
    dst = "__dst__"
    task_graph.add_node(src, weight=1e-9)
    task_graph.add_node(dst, weight=1e-9)
    for node in task_graph.nodes:
        if node == src or node == dst:
            continue
        elif task_graph.in_degree(node) == 0:
            task_graph.add_edge(src, node, weight=1e-9)
        elif task_graph.out_degree(node) == 0:
            task_graph.add_edge(node, dst, weight=1e-9)

def sanitize_schedule(schedule: Dict[Hashable, List[Task]]) -> Dict[Hashable, List[Task]]:
    """Removes dummy src and dst nodes from schedule.

    Args:
        schedule (Dict[Hashable, List[Task]]): [description]

    Returns:
        Dict[Hashable, List[Task]]: [description]
    """
    return {
        node: [task for task in tasks if task.name != "__src__" and task.name != "__dst__"]
        for node, tasks in schedule.items()
    }

def main():
    """Main function."""
    results = load_results()
    network = results.iterations[-1].best_network
    task_graph = results.iterations[-1].best_task_graph
    print(f"Network:\n{format_network(network)}")
    print(f"Task graph:\n{format_task_graph(task_graph)}")

    standardize_isntance(network, task_graph)

    schedule_heft = HeftScheduler().schedule(network, task_graph)
    schedule_cpop = CpopScheduler().schedule(network, task_graph)

    schedule_heft = sanitize_schedule(schedule_heft)
    schedule_cpop = sanitize_schedule(schedule_cpop)
    
    makespan_heft = max(task.end for tasks in schedule_heft.values() for task in tasks)
    makespan_cpop = max(task.end for tasks in schedule_cpop.values() for task in tasks)

    print(f"Algorithm 1 (makespan={makespan_heft:.2f}):\n{format_schedule(schedule_heft)}")
    print(f"Algorithm 2 (makespan={makespan_cpop:.2f}):\n{format_schedule(schedule_cpop)}")


if __name__ == "__main__":
    main()