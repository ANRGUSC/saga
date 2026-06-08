import pathlib
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, PrivateAttr, Field

from saga import Schedule, Scheduler, ScheduledTask, TaskGraph, Network, TaskGraphNode, NetworkNode, TaskGraphEdge
from saga.schedulers.cpop import upward_rank
import heapq





def compute_inspiring_effeciency(task_graph: TaskGraph, network: Network, time_window: float) -> Dict[str, float]:
    """Compute the inspiring efficiency of tasks in a task graph.
    A task's inspiring efficiency is the number of tasks expected to finish in a given time window.

    Args:
        task_graph (TaskGraph): The task graph to compute the inspiring efficiency for.
        network (Network): The network to compute the inspiring efficiency for.
        time_window (float): The time window to compute the inspiring efficiency for.

    Returns: 
       effeciency_ranks: A dict of task_name to inspiring efficiency, where a task's inspiring efficiency is the number of tasks expected to finish in the given time window.
    """
    average_network_speed = np.mean([node.speed for node in network.nodes])
    
    #the paper suggests using the average execution time of tasks of a certain type, but
    #because we don't have task types, we will just manually calculate excecution costs of 
    #of children tasks. This will be more computationally expensive, but can be easily optimized by task types.
    def compute_task_inspiring_ability(task) -> float:
        """Compute the inspiring ability of a task in a task graph.
        A task's inspiring ability is the total amount of tasks dependent on it that can be expected to finish in a given time window.
        
        Args:
            task (TaskGraphNode): The task to compute the inspiring ability for.
        Returns:
            float: The inspiring ability of the task.
        """
        count = 0
        # counter = 0
        time = 0
        current_gen = {task.name}
        next_gen = set()
        visited = set()
        while current_gen and time < time_window:
            cur_node = current_gen.pop()
            visited.add(cur_node)
            children = get_children(task_graph, cur_node)
            # next_gen.update({child.name for child in children if child.name not in visited})
            for child in children:
                if child.name not in visited:
                    visited.add(child.name)
                    time += child.cost / average_network_speed
                    if time >= time_window:
                        return count
                    count += 1
                    next_gen.add(child.name)
            if current_gen:
                continue
            else:   
                # visited.update(current_gen)
                current_gen = next_gen
                next_gen = set()
        return count
    effeciency_ranking = []
    # for task in task_graph.tasks:
    #     ability = compute_task_inspiring_ability(task)
    #     # counter += 1
    #     # heapq.heappush(effeciency_ranking, (-ability, counter, task.name))
    effeciency_ranks = {task.name: compute_task_inspiring_ability(task) for task in task_graph.tasks}
    return effeciency_ranks

        


def compute_inspiring_ability(task_graph: TaskGraph) -> dict[str, float]:
    """Compute the inspiring ability of tasks in a task graph.
    A task's inspiring ability is the total amount of tasks dependent on it.

    Args:
        task_graph (TaskGraph): The task graph to compute the inspiring ability for.

    Returns:
        ability_ranks: A dict of task_name to inspiring ability, where a task's inspiring ability is the total amount of tasks dependent on it.
        """
    
    topological_order = reversed(task_graph.topological_sort())
    reachable = {task: set() for task in task_graph.tasks}
    ability_ranking = []
    counter = 0
    for task in topological_order:
        for child in get_children(task_graph, task.name):
            reachable[task].add(child)
            reachable[task].update(reachable[child])
    ability_ranks = {task.name: (len(reachable[task])+1) for task in task_graph.tasks}
    return ability_ranks


def get_children(task_graph: TaskGraph, task_name: str) -> List[TaskGraphNode]:
    """Get the children of a task in a task graph.

    Args:
        task_graph (TaskGraph): The task graph to get the children from.
        task_name (str): The name of the task to get the children for.

    Returns:
        List[TaskGraphNode]: The child nodes of the task.
    """
    return [task_graph.get_task(edge.target)for edge in task_graph.out_edges(task_name) if edge.target not in {"__super_source__", "__super_sink__"}]







 
if __name__ == "__main__":
    import networkx as nx

    # Simple 1-node network with speed=1 → average_network_speed=1
    # so child.cost / average_network_speed == child.cost (weight)
    net = nx.Graph()
    net.add_node("v1", weight=1.0)
    from saga import Network
    network = Network.from_nx(net)

    def make_tg(nodes, edges):
        g = nx.DiGraph()
        for n in nodes:
            g.add_node(n, weight=1)
        for u, v in edges:
            g.add_edge(u, v, weight=1)
        return TaskGraph.from_nx(g)

    # --- compute_inspiring_efficiency tests ---

    # Test 1: linear chain A→B→C (all cost=1)
    # time_window=2.5: B(time=1), C(time=2) → 2
    # time_window=1.5: B(time=1), C(time=2 >= 1.5) → 1
    # time_window=0.5: B(time=1 >= 0.5) → 0
    tg_chain = make_tg(["A", "B", "C"], [("A", "B"), ("B", "C")])
    assert compute_inspiring_effeciency(tg_chain, network, 2.5)["A"] == 2, "chain window=2.5"
    assert compute_inspiring_effeciency(tg_chain, network, 1.5)["A"] == 1, "chain window=1.5"
    assert compute_inspiring_effeciency(tg_chain, network, 0.5)["A"] == 0, "chain window=0.5"
    print("Test 1 passed: linear chain")

    # Test 2: leaf task — no children, always 0
    tg_leaf = make_tg(["A"], [])
    assert compute_inspiring_effeciency(tg_leaf, network, 10.0)["A"] == 0, "leaf"
    print("Test 2 passed: leaf task")

    # Test 3: time_window=0 → always 0
    assert compute_inspiring_effeciency(tg_chain, network, 0.0)["A"] == 0, "window=0"
    print("Test 3 passed: time_window=0")

    # Test 4: diamond A→B, A→C, B→D, C→D — D counted once
    # For A: children B(time=1), C(time=2), then D(time=3) once → count=3
    tg_diamond = make_tg(["A", "B", "C", "D"], [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    assert compute_inspiring_effeciency(tg_diamond, network, 3.5)["A"] == 3, "diamond D once"
    # Narrow window cuts off before D: B(1), C(2), D would be 3 >= 2.5 → count=2
    assert compute_inspiring_effeciency(tg_diamond, network, 2.5)["A"] == 2, "diamond window=2.5"
    print("Test 4 passed: diamond (D counted once)")

    # Test 5: fan-out A→B, A→C, A→D (no convergence)
    # For A: B(1), C(2), D(3) → 3
    tg_fanout = make_tg(["A", "B", "C", "D"], [("A", "B"), ("A", "C"), ("A", "D")])
    assert compute_inspiring_effeciency(tg_fanout, network, 3.5)["A"] == 3, "fan-out"
    assert compute_inspiring_effeciency(tg_fanout, network, 1.5)["A"] == 1, "fan-out window=1.5"
    print("Test 5 passed: fan-out")

    # --- compute_inspiring_ability tests ---

    # Two roots, multiple merge/fan-out points:
    #
    # t_1 --> t_3 --> t_5 --> t_7
    # t_1 --> t_4 --> t_6 --> t_7
    # t_2 --> t_4
    # t_2 --> t_5
    # t_3 --> t_6
    #
    # expected:
    #   t_7: 1
    #   t_5: 2  (1 + t_7)
    #   t_6: 2  (1 + t_7)
    #   t_3: 5  (1 + t_5 + t_6) -- t_7 counted once per path
    #   t_4: 5  (1 + t_6 + t_5) -- same
    #   t_1: 11 (1 + t_3 + t_4)
    #   t_2: 11 (1 + t_4 + t_5)
    g = nx.DiGraph()
    for t in ["t_1", "t_2", "t_3", "t_4", "t_5", "t_6", "t_7", "t_8"]:
        g.add_node(t, weight=1)
    g.add_edge("t_1", "t_3", weight=1)
    g.add_edge("t_1", "t_4", weight=1)
    g.add_edge("t_2", "t_4", weight=1)
    g.add_edge("t_2", "t_5", weight=1)
    g.add_edge("t_3", "t_5", weight=1)
    g.add_edge("t_3", "t_6", weight=1)
    g.add_edge("t_4", "t_6", weight=1)
    g.add_edge("t_5", "t_7", weight=1)
    g.add_edge("t_6", "t_7", weight=1)
    g.add_edge("t_5", "t_8", weight=1)
    tg = TaskGraph.from_nx(g)
    result = compute_inspiring_ability(tg)
    for k, v in sorted(result.items()):
        print(f"{k}: {v}")

    # 