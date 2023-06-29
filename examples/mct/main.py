import itertools
from saga.common.olb import OLBScheduler, Task
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from dataclasses import dataclass

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from plotly.graph_objects import Figure 
import pathlib

thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    # create a diamond-task graph with random task weights between 0 and 1
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from([1, 2, 3, 4])
    task_graph.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
    
    for node in task_graph.nodes:
        task_graph.nodes[node]['weight'] = np.random.rand()
        
    for edge in task_graph.edges:
        task_graph.edges[edge]['weight'] = np.random.rand()

    # generate a fully-connected network with random node and edge weights between 0 and 1
    network = nx.Graph()
    network.add_nodes_from([1, 2, 3, 4])
    for node in network.nodes:
        network.nodes[node]['weight'] = np.random.rand()
        
    for (src, dst) in itertools.product(network.nodes, network.nodes):
        print(f"Adding edge {src} -> {dst}")
        if src != dst:
            network.add_edge(src, dst, weight=np.random.rand())
        else:
            network.add_edge(src, dst, weight=1e9)  # Large weight for self-connection

    scheduler = OLBScheduler()
    schedule = scheduler.schedule(network, task_graph)

    # draw the task graph
    ax: plt.Axes = draw_task_graph(task_graph)
    ax.set_title('Task Graph')
    fig = ax.get_figure()
    fig.savefig(thisdir / 'task_graph.png')

    # draw the network
    ax: plt.Axes = draw_network(network)
    ax.set_title('Network')
    fig = ax.get_figure()
    fig.savefig(thisdir / 'network.png')
    
    raw_schedule = scheduler.schedule(network, task_graph)

# Compute start and end times for each task
    schedule = {}
    for node, tasks in raw_schedule.items():
        start_time = 0
        schedule[node] = []
        for task in tasks:
            end_time = start_time + task_graph.nodes[task.name]['weight']
            schedule_task = Task(name=task.name, node=node, start=start_time, end=end_time)
            schedule[node].append(schedule_task)
            start_time = end_time  # The next task starts when the current one ends
    
    # draw the schedule
    fig: Figure = draw_gantt(schedule)
    fig.update_layout(title='Schedule')
    fig.write_image(str(thisdir / 'schedule.png'))

if __name__ == '__main__':
    main()
