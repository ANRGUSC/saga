from functools import lru_cache
import logging
import pathlib
from typing import Tuple
import networkx as nx

from saga import Network, Schedule, ScheduledTask, TaskGraph
from saga.schedulers import HeftScheduler, BruteForceScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

logging.basicConfig(level=logging.INFO)

thisdir = pathlib.Path(__file__).parent.absolute()
savedir = thisdir / 'outputs'
savedir.mkdir(exist_ok=True)

@lru_cache(maxsize=None)
def get_instance() -> Tuple[Network, TaskGraph]:
    network = nx.Graph()
    network.add_node("v_1", weight=1)
    network.add_node("v_2", weight=2)
    network.add_node("v_3", weight=2)
    network.add_edge("v_1", "v_2", weight=2)
    network.add_edge("v_1", "v_3", weight=2)
    network.add_edge("v_2", "v_3", weight=1)

    task_graph = nx.DiGraph()
    task_graph.add_node('t_1', weight=1)
    task_graph.add_node('t_2', weight=3)
    task_graph.add_node('t_3', weight=2)
    task_graph.add_node('t_4', weight=1)

    task_graph.add_edge('t_1', 't_2', weight=1)
    task_graph.add_edge('t_1', 't_3', weight=1)
    task_graph.add_edge('t_2', 't_4', weight=5)
    task_graph.add_edge('t_3', 't_4', weight=5)

    return Network.from_nx(network), TaskGraph.from_nx(task_graph)

def draw_instance(network: Network, task_graph: TaskGraph):
    logging.basicConfig(level=logging.INFO)
    ax = draw_task_graph(task_graph.graph, use_latex=False)
    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(str(savedir / 'task_graph.png'))

    ax = draw_network(network.graph, draw_colors=False, use_latex=False)
    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(str(savedir / 'network.png'))

def draw_schedule(schedule: Schedule, name: str, xmax: float | None = None):
    ax = draw_gantt(schedule.mapping, use_latex=False, xmax=xmax)
    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(str(savedir / f'{name}.png'))

def my_schedule() -> Schedule:
    network, task_graph = get_instance()
    task_name_map = {"t_1": "v_1", "t_2": "v_1", "t_3": "v_2", "t_4": "v_2"}
    schedule = Schedule(task_graph, network)
    for task_name, node_name in task_name_map.items():
        min_start_time = schedule.get_earliest_start_time(task_name, node_name)
        start_time = max(min_start_time, 0 if not schedule[node_name] else schedule[node_name][-1].end)
        task = task_graph.get_task(task_name)
        node = network.get_node(node_name)
        new_task = ScheduledTask(
            name=task_name,
            node=node_name,
            start=start_time,
            end=start_time + task.cost / node.speed
        )
        schedule.add_task(new_task)
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
        sched.makespan
        for sched in [schedule, heft_sched, bf_sched]
    )
    print(f"Max makespan: {max_makespan}")
    draw_schedule(schedule, 'gantt_scaled', xmax=max_makespan)
    draw_schedule(heft_sched, 'heft_gantt_scaled', xmax=max_makespan)
    draw_schedule(bf_sched, 'brute_force_gantt_scaled', xmax=max_makespan)

if __name__ == '__main__':
    main()
