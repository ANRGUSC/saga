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
    task_graph.add_node('Z', weight=rand_value(1, 1/3))
    for node in inner_nodes:
        task_graph.add_edge('A', node, weight=rand_value(1, 1/3))
        task_graph.add_edge(node, 'Z', weight=rand_value(10, 10/3))

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
    task_graph.add_node('Z', weight=1)

    cost = rand_value(10, 10/3)
    task_graph.add_node('B', weight=cost)
    task_graph.add_node('F', weight=cost)
    
    task_graph.add_edge('A', 'B', weight=1)
    task_graph.add_edge('B', 'Z', weight=1)

    task_graph.add_edge('A', 'F', weight=rand_value(100, 0))
    task_graph.add_edge('F', 'Z', weight=1)

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
    ax = draw_gantt(cpop_schedule)
    ax.set_title('CPoP Schedule')
    plt.savefig(save_dir / 'cpop_schedule.png')

    ax = draw_gantt(heft_schedule)
    ax.set_title('HEFT Schedule')
    plt.savefig(save_dir / 'heft_schedule.png')

    ax = draw_network(network)
    ax.set_title('Network')
    plt.savefig(save_dir / 'network.png')

    ax = draw_task_graph(task_graph)
    ax.set_title('Task Graph')
    plt.savefig(save_dir / 'task_graph.png')

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
        title='CPoP vs HEFT Makespan',
        labels={'variable': 'Scheduler', 'value': 'Makespan'},
    )
    # make boxes black/grey
    fig.update_traces(marker=dict(color='black', size=3))
    plt.tight_layout()
    fig.write_image(save_dir / 'cpop_vs_heft_boxplot.png')

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
    ax.set_title('CPoP Schedule')
    plt.savefig(save_dir / 'cpop_schedule.png')

    ax = draw_gantt(heft_schedule)
    ax.set_title('HEFT Schedule')
    plt.savefig(save_dir / 'heft_schedule.png')

    ax = draw_network(network)
    ax.set_title('Network')
    plt.savefig(save_dir / 'network.png')

    ax = draw_task_graph(task_graph)
    ax.set_title('Task Graph')
    plt.savefig(save_dir / 'task_graph.png')

    # return

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
        title='CPoP vs HEFT Makespan',
        labels={'variable': 'Scheduler', 'value': 'Makespan'},
    )
    # make boxes black/grey
    fig.update_traces(marker=dict(color='black', size=3))
    plt.tight_layout()
    fig.write_image(save_dir / 'cpop_vs_heft_boxplot.png')

    # makespan ratio
    df['Ratio'] = df['HEFT'] / df['CPoP']

    avg_ratio = df['Ratio'].mean()
    std_ratio = df['Ratio'].std()

    print(f'HEFT/CPoP Ratio: {avg_ratio} ± {std_ratio}')


def main():
    bad_cpop()
    bad_heft()

if __name__ == '__main__':
    main()