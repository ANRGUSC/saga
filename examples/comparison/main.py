from itertools import product
from typing import Dict

import numpy as np
import pandas as pd
from saga.base import Scheduler
from saga.common.heft import HeftScheduler
from saga.common.minmin import MinMinScheduler
from saga.common.maxmin import MaxMinScheduler
from saga.common.duplex import DuplexScheduler
from saga.utils.draw import draw_task_graph, draw_network, draw_gantt

import pathlib
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px

thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    task_graph = nx.DiGraph() 
    task_graph.add_node('A', weight=1)
    task_graph.add_node('B', weight=7)
    task_graph.add_node('C', weight=5)
    task_graph.add_node('D', weight=1)
    task_graph.add_node('E', weight=1)

    task_graph.add_edge('A', 'B', weight=1)
    task_graph.add_edge('A', 'C', weight=5)
    task_graph.add_edge('A', 'D', weight=1)
    task_graph.add_edge('B', 'E', weight=5)
    task_graph.add_edge('C', 'E', weight=1)
    task_graph.add_edge('D', 'E', weight=1)

    network = nx.Graph()
    network.add_node(1, weight=1)
    network.add_node(2, weight=1)
    network.add_node(3, weight=1)
    network.add_node(4, weight=1)

    for (src, dst) in product(network.nodes, network.nodes):
        if src != dst:
            network.add_edge(src, dst, weight=1)
        else:
            network.add_edge(src, dst, weight=1e9) # basically infinity

    schedulers: Dict[str, Scheduler] = {
        'HEFT': HeftScheduler(),
        'MinMin': MinMinScheduler(),
        'MaxMin': MaxMinScheduler(),
        'Duplex': DuplexScheduler(),
    }

    rows = []
    for name, scheduler in schedulers.items():
        savedir = thisdir / 'results' / name
        savedir.mkdir(exist_ok=True, parents=True)
        schedule = scheduler.schedule(network, task_graph)

        makespan = max([task.end for node, tasks in schedule.items() for task in tasks])
        rows.append([name, makespan])

        ax: plt.Axes = draw_task_graph(task_graph)
        fig = ax.get_figure()
        fig.savefig(str(savedir / 'task_graph.png'))

        ax: plt.Axes = draw_network(network)
        fig = ax.get_figure()
        fig.savefig(str(savedir / 'network.png'))

        fig = draw_gantt(schedule) # plotly figure
        fig.write_image(str(savedir / 'schedule.png'))

    df = pd.DataFrame(rows, columns=['Scheduler', 'Makespan'])
    fig = px.bar(df, x='Scheduler', y='Makespan')
    fig.write_image(str(thisdir / 'results' / 'comparison.png'))
    print(df)

if __name__ == '__main__':
    main()