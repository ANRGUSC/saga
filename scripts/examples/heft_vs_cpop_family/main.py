from itertools import combinations
import networkx as nx
import numpy as np
from saga.schedulers.cpop import CpopScheduler
from saga.schedulers.heft import HeftScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
import pathlib
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

thisdir = pathlib.Path(__file__).parent.absolute()


plt.rcParams.update({
    # 'font.size': 20,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

FILETYPE = "pdf" # pdf or png

def rand_value(mean: float, std: float) -> float:
    if std == 0:
        return mean
    return max(1e-9, np.random.normal(mean, std))

def get_bad_cpop_example():
    """
    
    Scenario: fork/join type graph where fork has low communication and join has higher communication
              the fastest node has one weak link
    Expected: HEFT should perform better than CPoP
    """
    task_graph = nx.DiGraph()
    inner_nodes = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    task_graph.add_node('A', weight=rand_value(1, 1/3))
    for node in inner_nodes:
        task_graph.add_node(node, weight=rand_value(1, 1/3))
    task_graph.add_node('K', weight=rand_value(1, 1/3))
    for node in inner_nodes:
        task_graph.add_edge('A', node, weight=rand_value(1, 1/3))
        task_graph.add_edge(node, 'K', weight=rand_value(10, 10/3))

    network = nx.Graph()
    network.add_node(1, weight=rand_value(3, 0))
    network.add_node(2, weight=rand_value(1, 1/3))
    network.add_node(3, weight=rand_value(1, 1/3))
    network.add_node(4, weight=rand_value(1, 1/3))

    for node in network.nodes:
        network.add_edge(node, node, weight=1e9)

    for node, other_node in combinations(network.nodes, 2):
        if node == 1 and other_node == 2:
            network.add_edge(node, other_node, weight=rand_value(1, 1/3))
        else:
            network.add_edge(node, other_node, weight=rand_value(10, 5/3))

    return network, task_graph


def get_bad_heft_example():
    """
    Scenario: Fork/join type DAG where the first chain has a higher communication cost to start
              homogeneous network
    Expected: CPoP should perform better than HEFT
    """

    task_graph = nx.DiGraph()
    task_graph.add_node('A', weight=1)
    task_graph.add_node('D', weight=1)

    cost = rand_value(10, 10/3)
    task_graph.add_node('B', weight=cost)
    task_graph.add_node('C', weight=cost)
    
    task_graph.add_edge('A', 'B', weight=1)
    task_graph.add_edge('B', 'D', weight=1)

    task_graph.add_edge('A', 'C', weight=rand_value(100, 100/3))
    task_graph.add_edge('C', 'D', weight=1)

    network = nx.Graph()
    network.add_node(1, weight=rand_value(1, 0))
    network.add_node(2, weight=rand_value(1, 0))
    network.add_node(3, weight=rand_value(1, 0))

    for node in network.nodes:
        network.add_edge(node, node, weight=1e9)

    for node, other_node in combinations(network.nodes, 2):
        network.add_edge(node, other_node, weight=rand_value(1, 0))

    return network, task_graph

def bad_cpop():
    scheduler_cpop = CpopScheduler()
    scheduler_heft = HeftScheduler()
    num_samples = 1000

    save_dir = thisdir / 'bad_cpop'
    save_dir.mkdir(exist_ok=True, parents=True)

    # draw schedule for one example
    network, task_graph = get_bad_cpop_example()
    cpop_schedule = scheduler_cpop.schedule(network, task_graph)
    heft_schedule = scheduler_heft.schedule(network, task_graph)
    ax = draw_gantt(cpop_schedule, use_latex=True)
    plt.savefig(save_dir / f'cpop_schedule.{FILETYPE}')

    ax = draw_gantt(heft_schedule, use_latex=True)
    plt.savefig(save_dir / f'heft_schedule.{FILETYPE}')

    ax = draw_network(network, draw_colors=False, use_latex=True)
    plt.savefig(save_dir / f'network.{FILETYPE}')

    # remove tasks C through I
    task_graph.remove_nodes_from(['C', 'D', 'E', 'F', 'G', 'H', 'I'])
    # set B and J label to \mathcal{N}(10, 10/3)
    for node in task_graph.nodes:
        task_graph.nodes[node]['label'] = r'\mathcal{N}(1, \frac{1}{3})'

    task_graph.edges['A', 'B']['label'] = r'\mathcal{N}(1, \frac{1}{3})'
    task_graph.edges['A', 'J']['label'] = r'\mathcal{N}(1, \frac{1}{3})'
    task_graph.edges['B', 'K']['label'] = r'\mathcal{N}(10, \frac{10}{3})'
    task_graph.edges['J', 'K']['label'] = r'\mathcal{N}(10, \frac{10}{3})'

    task_graph_pos = {'A': (0, 0), 'B': (-1, -1), 'J': (1, -1), 'K': (0, -2)}
    ax = draw_task_graph(task_graph, use_latex=True, pos=task_graph_pos, figsize=(8, 6))
    # draw a ... between B and J
    ax.text(0, -1, r'$\ldots$', fontsize=20, ha='center', va='center')
    plt.savefig(save_dir / f'task_graph.{FILETYPE}')

    # Test Bad CPoP Examples
    cpop_makespans = []
    heft_makespans = []
    for _ in range(num_samples):
        network, task_graph = get_bad_cpop_example()
        cpop_schedule = scheduler_cpop.schedule(network, task_graph)
        heft_schedule = scheduler_heft.schedule(network, task_graph)

        cpop_makespan = max(task.end for tasks in cpop_schedule.values() for task in tasks)
        heft_makespan = max(task.end for tasks in heft_schedule.values() for task in tasks)

        cpop_makespans.append(cpop_makespan)
        heft_makespans.append(heft_makespan)

    df = pd.DataFrame({
        'CPoP': cpop_makespans,
        'HEFT': heft_makespans
    })

    # boxplot
    fig = px.box(
        df, 
        y=['CPoP', 'HEFT'],
        template='plotly_white',
        labels={'variable': 'Scheduler', 'value': 'Makespan'},
    )
    # make boxes black/grey
    fig.update_traces(marker=dict(color='black', size=3))
    plt.tight_layout()
    fig.write_image(save_dir / f'cpop_vs_heft_boxplot.{FILETYPE}')

    # makespan ratio
    df['Ratio'] = df['CPoP'] / df['HEFT']
    
    avg_ratio = df['Ratio'].mean()
    std_ratio = df['Ratio'].std()

    print(f'CPoP/HEFT Ratio: {avg_ratio} ± {std_ratio}')


def bad_heft():
    scheduler_cpop = CpopScheduler()
    scheduler_heft = HeftScheduler()
    num_samples = 1000

    save_dir = thisdir / 'bad_heft'
    save_dir.mkdir(exist_ok=True, parents=True)

    # draw schedule for one example
    network, task_graph = get_bad_heft_example()
    cpop_schedule = scheduler_cpop.schedule(network, task_graph)
    heft_schedule = scheduler_heft.schedule(network, task_graph)

    ax = draw_gantt(cpop_schedule)
    plt.savefig(save_dir / f'cpop_schedule.{FILETYPE}')

    ax = draw_gantt(heft_schedule)
    plt.savefig(save_dir / f'heft_schedule.{FILETYPE}')



    ax = draw_network(network, draw_colors=False, use_latex=True)
    plt.savefig(save_dir / f'network.{FILETYPE}')

    # set B and C label to \mathcal{N}(10, 10/3)
    for node in task_graph.nodes:
        if node in ['B', 'C']:
            task_graph.nodes[node]['label'] = r'\mathcal{N}(10, \frac{10}{3})'

    task_graph.edges['A', 'C']['label'] = r'\mathcal{N}(100, \frac{100}{3})'
    # A on top, B and C on the same level, D on the bottom
    task_graph_pos = {'A': (0, 0), 'B': (-1, -1), 'C': (1, -1), 'D': (0, -2)}
    ax = draw_task_graph(task_graph, use_latex=True, pos=task_graph_pos, figsize=(8, 6))
    plt.savefig(save_dir / f'task_graph.{FILETYPE}')

    # Test Bad HEFT Examples
    cpop_makespans = []
    heft_makespans = []
    for _ in range(num_samples):
        network, task_graph = get_bad_heft_example()
        cpop_schedule = scheduler_cpop.schedule(network, task_graph)
        heft_schedule = scheduler_heft.schedule(network, task_graph)

        cpop_makespan = max(task.end for tasks in cpop_schedule.values() for task in tasks)
        heft_makespan = max(task.end for tasks in heft_schedule.values() for task in tasks)

        cpop_makespans.append(cpop_makespan)
        heft_makespans.append(heft_makespan)

    df = pd.DataFrame({
        'CPoP': cpop_makespans,
        'HEFT': heft_makespans
    })

    # boxplot
    fig = px.box(
        df, 
        y=['CPoP', 'HEFT'],
        template='plotly_white',
        labels={'variable': 'Scheduler', 'value': 'Makespan'},
    )
    # make boxes black/grey
    fig.update_traces(marker=dict(color='black', size=3))
    plt.tight_layout()
    fig.write_image(save_dir / f'cpop_vs_heft_boxplot.{FILETYPE}')

    # makespan ratio
    df['Ratio'] = df['HEFT'] / df['CPoP']

    avg_ratio = df['Ratio'].mean()
    std_ratio = df['Ratio'].std()

    print(f'HEFT/CPoP Ratio: {avg_ratio} ± {std_ratio}')

def paper_example_bad_cpop():
    """Approximate the example from the paper where CPoP performs worse than HEFT
    
    The example in the paper is rounded to the nearest 0.1, so the weights are not exactly the same.
    The idea is the same though and actually the makespan ratio here is even worse!
    """
    scheduler_cpop = CpopScheduler()
    scheduler_heft = HeftScheduler()

    network = nx.Graph()
    network.add_node(1, weight=0.4)
    network.add_node(2, weight=0.8)
    network.add_node(3, weight=1e-9)
    network.add_edge(1, 2, weight=0.1)
    network.add_edge(2, 3, weight=0.3)
    network.add_edge(1, 3, weight=0.4)

    for node in network.nodes:
        network.add_edge(node, node, weight=1e9)

    task_graph = nx.DiGraph()
    task_graph.add_node('A', weight=0.8)
    task_graph.add_node('B', weight=0.9)
    task_graph.add_node('C', weight=0.6)
    task_graph.add_edge('A', 'C', weight=0.7)
    task_graph.add_edge('B', 'C', weight=0.2)

    # # add fake source and sink
    task_graph.add_node('__source__', weight=1e-9)
    task_graph.add_node('__sink__', weight=1e-9)
    task_graph.add_edge('__source__', 'A', weight=1e-9)
    task_graph.add_edge('__source__', 'B', weight=1e-9)
    task_graph.add_edge('C', '__sink__', weight=1e-9)

    cpop_schedule = scheduler_cpop.schedule(network, task_graph)
    heft_schedule = scheduler_heft.schedule(network, task_graph)

    savedir = thisdir / 'paper_example_bad_cpop'
    savedir.mkdir(exist_ok=True, parents=True)

    ax = draw_gantt(cpop_schedule, use_latex=True)
    plt.savefig(savedir / f'cpop_schedule.{FILETYPE}')

    ax = draw_gantt(heft_schedule, use_latex=True)
    plt.savefig(savedir / f'heft_schedule.{FILETYPE}')

    ax = draw_network(network, draw_colors=False)
    plt.savefig(savedir / f'network.{FILETYPE}')

    ax = draw_task_graph(task_graph)
    plt.savefig(savedir / f'task_graph.{FILETYPE}')

    cpop_makespan = max(task.end for tasks in cpop_schedule.values() for task in tasks)
    heft_makespan = max(task.end for tasks in heft_schedule.values() for task in tasks)

    print(f'CPoP Makespan: {cpop_makespan}')
    print(f'HEFT Makespan: {heft_makespan}')
    
    print(f'Makespan Ratio: {cpop_makespan / heft_makespan}')

def paper_example_bad_heft():
    """Approximate the example from the paper where HEFT performs worse than CPoP
    
    The example in the paper is rounded to the nearest 0.1, so the weights are not exactly the same.
    The idea is the same though.
    """
    scheduler_cpop = CpopScheduler()
    scheduler_heft = HeftScheduler()

    network = nx.Graph()
    network.add_node(1, weight=0.4)
    network.add_node(2, weight=0.7)
    network.add_node(3, weight=0.5)
    network.add_edge(1, 2, weight=0.6)
    network.add_edge(2, 3, weight=0.1)
    network.add_edge(1, 3, weight=0.3)

    for node in network.nodes:
        network.add_edge(node, node, weight=1e9)

    task_graph = nx.DiGraph()
    task_graph.add_node('A', weight=0.8)
    task_graph.add_node('B', weight=0.02)
    task_graph.add_node('C', weight=0.8)
    task_graph.add_edge('B', 'C', weight=0.8)

    # # add fake source and sink
    task_graph.add_node('__source__', weight=1e-9)
    task_graph.add_node('__sink__', weight=1e-9)
    task_graph.add_edge('__source__', 'A', weight=1e-9)
    task_graph.add_edge('__source__', 'B', weight=1e-9)
    task_graph.add_edge('A', '__sink__', weight=1e-9)
    task_graph.add_edge('C', '__sink__', weight=1e-9)

    cpop_schedule = scheduler_cpop.schedule(network, task_graph)
    heft_schedule = scheduler_heft.schedule(network, task_graph)

    savedir = thisdir / 'paper_example_bad_heft'
    savedir.mkdir(exist_ok=True, parents=True)

    ax = draw_gantt(cpop_schedule, use_latex=True)
    plt.savefig(savedir / f'cpop_schedule.{FILETYPE}')

    ax = draw_gantt(heft_schedule, use_latex=True)
    plt.savefig(savedir / f'heft_schedule.{FILETYPE}')

    # set B and C label to \mathcal{N}(10, 10/3)
    for node in task_graph.nodes:
        if node in ['B', 'C']:
            task_graph.nodes[node]['label'] = r'\mathcal{N}(10, \frac{10}{3})'

    task_graph.edges['B', 'C']['label'] = r'\mathcal{N}(100, \frac{100}{3})'
    # A on top, B and C on the same level
    task_graph_pos = {'A': (0, 0), 'B': (-1, -1), 'C': (1, -1)}
    
    ax = draw_network(network, draw_colors=False, use_latex=True)
    plt.savefig(savedir / f'network.{FILETYPE}')

    ax = draw_task_graph(task_graph, use_latex=True, pos=task_graph_pos)
    plt.savefig(savedir / f'task_graph.{FILETYPE}')

    cpop_makespan = max(task.end for tasks in cpop_schedule.values() for task in tasks)
    heft_makespan = max(task.end for tasks in heft_schedule.values() for task in tasks)

    print(f'CPoP Makespan: {cpop_makespan}')
    print(f'HEFT Makespan: {heft_makespan}')

    print(f'Makespan Ratio: {heft_makespan / cpop_makespan}')

def main():
    # paper_example_bad_cpop()
    # paper_example_bad_heft()
    
    bad_cpop()
    bad_heft()

if __name__ == '__main__':
    main()