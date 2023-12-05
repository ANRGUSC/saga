"""Post-experiment LLM explanation."""
import pathlib
import pickle
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import pyperclip
from simulated_annealing import SimulatedAnnealing

from saga.scheduler import Task

thisdir = pathlib.Path(__file__).parent.absolute()


def format_schedule(schedule: Dict[Hashable, List[Task]]) -> str:
    """Converts schedule to string representation.

    Args:
        schedule (Dict[Hashable, List[Task]]): schedule of tasks to execute on each node

    Returns:
        str: string representation of schedule
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
        task_graph (nx.DiGraph): task graph

    Returns:
        str: string representation of task graph
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
        network (nx.Graph): network

    Returns:
        str: string representation of network
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


def sanitize_schedule(schedule: Dict[Hashable, List[Task]]) -> Dict[Hashable, List[Task]]:
    """Removes dummy src and dst nodes from schedule.

    Args:
        schedule (Dict[Hashable, List[Task]]): schedule of tasks to execute on each node

    Returns:
        Dict[Hashable, List[Task]]: sanitized schedule
    """
    return {
        node: [task for task in tasks if task.name != "__src__" and task.name != "__dst__"]
        for node, tasks in schedule.items()
    }

def get_top_results(results: SimulatedAnnealing,
                    num: int,
                    max_iter: int = -1) -> List[Tuple[nx.Graph, nx.DiGraph,
                                                      Dict[Hashable, List[Task]],
                                                      Dict[Hashable, List[Task]]]]:
    """Returns top x results.

    Args:
        results (SimulatedAnnealing): results
        num (int): number of results to return
        max_iter (int, optional): maximum number of iterations to consider. Defaults to -1 (all iterations).

    Returns:
        List[SimulatedAnnealingIteration]: [description]
    """
    assert (num > 0 and num >= max_iter and  max_iter >= -1 and
            max_iter <= len(results.iterations) and num <= len(results.iterations))
    top_iterations = sorted(results.iterations[:max_iter], key=lambda x: x.neighbor_energy)[-num:]
    return [
        (iteration.neighbor_network, iteration.neighbor_task_graph,
         iteration.neighbor_schedule, iteration.neighbor_base_schedule)
        for iteration in top_iterations
    ]

def format_instance(network: nx.Graph,
                    task_graph: nx.DiGraph,
                    schedule_1: Dict[Hashable, List[Task]],
                    schedule_2: Dict[Hashable, List[Task]]) -> str:
    """Formats instance for prompt.

    Args:
        network (nx.Graph): network
        task_graph (nx.DiGraph): task graph
        schedule_1 (Dict[Hashable, List[Task]]): schedule of tasks to execute on each node for algorithm 1
        schedule_2 (Dict[Hashable, List[Task]]): schedule of tasks to execute on each node for algorithm 2

    Returns:
        str: formatted instance
    """
    lines = []

    lines.append(f"Network:\n{format_network(network)}")
    lines.append(f"Task graph:\n{format_task_graph(task_graph)}")

    schedule_1 = sanitize_schedule(schedule_1)
    schedule_2 = sanitize_schedule(schedule_2)

    makespan_1 = max(task.end for tasks in schedule_1.values() for task in tasks)
    makespan_2 = max(task.end for tasks in schedule_2.values() for task in tasks)

    lines.append(f"Algorithm 1 (makespan={makespan_1:.2f}):\n{format_schedule(schedule_1)}")
    lines.append(f"Algorithm 2 (makespan={makespan_2:.2f}):\n{format_schedule(schedule_2)}")

    return "\n".join(lines)

def main(): # pylint: disable=too-many-locals
    """Main function."""
    algorithm_1, algorithm_2 = "HEFT", "CPOP"
    results_1: SimulatedAnnealing = pickle.loads(
        (thisdir / "results" / algorithm_1 / f"{algorithm_2}.pkl").read_bytes()
    )
    results_2: SimulatedAnnealing = pickle.loads(
        (thisdir / "results" / algorithm_2 / f"{algorithm_1}.pkl").read_bytes()
    )
    top_results_1 = get_top_results(results_1, num=3)
    top_results_2 = get_top_results(results_2, num=3)

    prompt_lines = [
        "Generally both Algorithm 1 and Algorithm 2 are both good scheduling algorithms "
        "(attempting to minimize makespan). Given the following instances and the schedules "
        "produced by Algorithm 1 and 2, can you hypothesize why one algorithm greatly "
        "outperforms the other for these instance and what that might imply about Algorithm "
        "1 and/or 2?\n",
        "Model: Task Graphs are directed acyclic graphs. Tasks must finish executing and "
        "send their data to dependent tasks before dependent tasks can start executing. "
        "For example if Task A is a dependent of Task B with data_size=d and Task B "
        "finishes executing on node 0 at time t. Then Task A cannot start executing on node "
        "1 until time t + d/s where s is the comm_speed between node 0 and node 1.\n\n"
    ]

    instance_count = 1
    for network, task_graph, schedule, base_schedule in top_results_1:
        prompt_lines.append(f"Instance {instance_count}:")
        prompt_lines.append(format_instance(network, task_graph, base_schedule, schedule))
        instance_count += 1

    for network, task_graph, schedule, base_schedule in top_results_2:
        prompt_lines.append(f"Instance {instance_count}:")
        prompt_lines.append(format_instance(network, task_graph, schedule, base_schedule))
        instance_count += 1

    mid_result = results_2.iterations[int(len(results_2.iterations) * 0.5) - 1]

    prompt_lines.append("Here is a random instance from the middle of the simulated annealing run. "
                        "Based on the information above, which algorithm do you think will perform "
                        "better on this instance?\n")
    prompt_lines.append(f"Network:\n{format_network(mid_result.best_network)}")
    prompt_lines.append(f"Task graph:\n{format_task_graph(mid_result.best_task_graph)}")

    prompt = "\n".join(prompt_lines)
    print(prompt)

    pyperclip.copy(prompt)

    makespan_1 = max(task.end for tasks in mid_result.best_schedule.values() for task in tasks)
    makespan_2 = max(task.end for tasks in mid_result.best_base_schedule.values() for task in tasks)

    print(f"Algorithm 1 makespan: {makespan_1:.2f}")
    print(f"Algorithm 2 makespan: {makespan_2:.2f}")


if __name__ == "__main__":
    main()
