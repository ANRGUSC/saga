from functools import lru_cache
import logging
import pathlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from saga.scheduler import Task
from saga.schedulers import HeftScheduler, BruteForceScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

logging.basicConfig(level=logging.INFO)

thisdir = pathlib.Path(__file__).parent.absolute()


plt.rcParams.update({
    # 'font.size': 20,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

@lru_cache(maxsize=None)
def get_instance() -> Tuple[nx.Graph, nx.DiGraph]:
    network = nx.Graph()
    network.add_node("v_1", weight=1)
    network.add_node("v_2", weight=2)
    network.add_node("v_3", weight=2)
    network.add_edge("v_1", "v_2", weight=2)
    network.add_edge("v_1", "v_3", weight=2)
    network.add_edge("v_2", "v_3", weight=1)
    # add self loops
    network.add_edge("v_1", "v_1", weight=1e9)
    network.add_edge("v_2", "v_2", weight=1e9)
    network.add_edge("v_3", "v_3", weight=1e9)

    task_graph = nx.DiGraph()
    task_graph.add_node('t_1', weight=1)
    task_graph.add_node('t_2', weight=3)
    task_graph.add_node('t_3', weight=2)
    task_graph.add_node('t_4', weight=1)

    task_graph.add_edge('t_1', 't_2', weight=1)
    task_graph.add_edge('t_1', 't_3', weight=1)
    task_graph.add_edge('t_2', 't_4', weight=5)
    task_graph.add_edge('t_3', 't_4', weight=5)

    return network, task_graph

def draw_instance(network: nx.Graph, task_graph: nx.DiGraph):
    logging.basicConfig(level=logging.INFO)
    ax: plt.Axes = draw_task_graph(task_graph, use_latex=True)
    ax.get_figure().savefig(str(thisdir / 'task_graph.png'))

    ax: plt.Figure = draw_network(network, draw_colors=False, use_latex=True)
    ax.get_figure().savefig(str(thisdir / 'network.png'))

def draw_schedule(schedule: Dict[str, List[Task]], name: str, xmax: float = None):
    ax: plt.Axes = draw_gantt(schedule, use_latex=True, xmax=xmax)
    ax.get_figure().savefig(str(thisdir / f'{name}.png'))

def my_schedule() -> Dict[str, List[Task]]:
    network, task_graph = get_instance()
    task_name_map = {"t_1": "v_1", "t_2": "v_1", "t_3": "v_2", "t_4": "v_2"}
    task_map: Dict[str, Task] = {}
    schedule = {node: [] for node in network.nodes}
    for task_name, node_name in task_name_map.items():
        print(f"Scheduling {task_name} on {node_name}")
        min_start_time = max(
            [
                task_map[pred].end + (
                    task_graph.edges[pred, task_name]["weight"]
                    / network.edges[task_map[pred].node, node_name]["weight"]
                )
                for pred in task_graph.predecessors(task_name)
            ],
            default=0
        )
        start_time = max(min_start_time, 0 if not schedule[node_name] else schedule[node_name][-1].end)
        task = Task(
            name=task_name,
            node=node_name,
            start=start_time,
            end=start_time + task_graph.nodes[task_name]["weight"] / network.nodes[node_name]["weight"]
        )
        schedule[node_name].append(task)
        task_map[task_name] = task

    return schedule

def heft_schedule():
    network, task_graph = get_instance()
    scheduler = HeftScheduler()
    schedule = scheduler.schedule(network, task_graph)
    return schedule

def brute_force_schedule():
    network, task_graph = get_instance()
    scheduler = BruteForceScheduler()
    schedule = scheduler.schedule(network, task_graph)
    return schedule

def main():
    draw_instance(*get_instance())
    schedule = my_schedule()
    draw_schedule(schedule, 'gantt')
    heft_sched = heft_schedule()
    draw_schedule(heft_sched, 'heft_gantt')
    bf_sched = brute_force_schedule()
    draw_schedule(bf_sched, 'brute_force_gantt')

    max_makespan = max(
        task.end 
        for sched in [schedule, heft_sched, bf_sched]
        for tasks in sched.values()
        for task in tasks
    )
    draw_schedule(schedule, 'gantt_scaled', xmax=max_makespan)
    draw_schedule(heft_sched, 'heft_gantt_scaled', xmax=max_makespan)
    draw_schedule(bf_sched, 'brute_force_gantt_scaled', xmax=max_makespan)

if __name__ == '__main__':
    main()
