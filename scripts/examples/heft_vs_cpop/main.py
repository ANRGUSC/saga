import networkx as nx
from saga.schedulers.cpop import CpopScheduler
from saga.schedulers.heft import HeftScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
import pathlib
import matplotlib.pyplot as plt

thisdir = pathlib.Path(__file__).parent.absolute()

font_size = 20


def cpop_vs_heft():
    cpop_scheduler = CpopScheduler()
    heft_scheduler = HeftScheduler()

    # scheduler=CPoP, baseline=HEFT
    network = nx.Graph()
    network.add_node(1, weight=0.37)
    network.add_node(2, weight=0.76)
    network.add_node(3, weight=0.04)

    network.add_edge(1, 2, weight=0.12)
    network.add_edge(1, 3, weight=0.41)

    network.add_edge(2, 3, weight=0.32)

    # add self loops
    network.add_edge(1, 1, weight=1e9)
    network.add_edge(2, 2, weight=1e9)
    network.add_edge(3, 3, weight=1e9)

    task_graph = nx.DiGraph()
    task_graph.add_node('__src__', weight=1e-9)
    task_graph.add_node('A', weight=0.76)
    task_graph.add_node('B', weight=0.94)
    task_graph.add_node('C', weight=0.57)
    task_graph.add_node('__dst__', weight=1e-9)

    task_graph.add_edge('__src__', 'A', weight=1e-9)
    task_graph.add_edge('__src__', 'B', weight=1e-9)
    task_graph.add_edge('A', 'C', weight=0.68)
    task_graph.add_edge('B', 'C', weight=0.21)
    task_graph.add_edge('C', '__dst__', weight=1e-9)

    cpop_schedule = cpop_scheduler.schedule(network, task_graph)
    heft_schedule = heft_scheduler.schedule(network, task_graph)

    cpop_makespan = max([0 if not tasks else tasks[-1].end for tasks in cpop_schedule.values()])
    heft_makespan = max([0 if not tasks else tasks[-1].end for tasks in heft_schedule.values()])

    print(f'CPoP makespan: {cpop_makespan}')
    print(f'HEFT makespan: {heft_makespan}')
    print(f"Makespan Ratio: {cpop_makespan/heft_makespan}")

    # task graph with __src__ and __dst__ nodes removed
    clean_task_graph = task_graph.copy()
    clean_task_graph.remove_node('__src__')
    clean_task_graph.remove_node('__dst__')

    savepath = thisdir / 'cpop_vs_heft'
    savepath.mkdir(exist_ok=True)
    axis = draw_task_graph(
        clean_task_graph, use_latex=True,
        font_size=font_size,
        weight_font_size=font_size
    )
    axis.get_figure().savefig(savepath / 'task_graph.pdf')
    plt.close(axis.get_figure())
    
    axis = draw_network(
        network, draw_colors=False, use_latex=True,
        font_size=font_size,
        weight_font_size=font_size
    )
    axis.get_figure().savefig(savepath / 'network.pdf')
    
    max_makespan = max(heft_makespan, cpop_makespan)
    axis = draw_gantt(
        cpop_schedule, use_latex=True, xmax=max_makespan,
        font_size=font_size,
        tick_font_size=font_size
    )
    axis.get_figure().savefig(savepath / 'cpop_gantt.pdf')
    plt.close(axis.get_figure())

    axis = draw_gantt(
        heft_schedule, use_latex=True, xmax=max_makespan,
        font_size=font_size,
        tick_font_size=font_size
    )
    axis.get_figure().savefig(savepath / 'heft_gantt.pdf')
    plt.close(axis.get_figure())


def heft_vs_cpop():
    cpop_scheduler = CpopScheduler()
    heft_scheduler = HeftScheduler()

    # scheduler=CPoP, baseline=HEFT
    network = nx.Graph()
    network.add_node(1, weight=0.37)
    network.add_node(2, weight=0.66)
    network.add_node(3, weight=0.52)

    network.add_edge(1, 2, weight=0.55)
    network.add_edge(1, 3, weight=0.32)
    network.add_edge(2, 3, weight=0.12)

    # add self loops
    network.add_edge(1, 1, weight=1e9)
    network.add_edge(2, 2, weight=1e9)
    network.add_edge(3, 3, weight=1e9)

    task_graph = nx.DiGraph()
    task_graph.add_node('__src__', weight=1e-9)
    task_graph.add_node('A', weight=0.84)
    task_graph.add_node('B', weight=0.03)
    task_graph.add_node('C', weight=0.77)
    task_graph.add_node('__dst__', weight=1e-9)

    task_graph.add_edge('__src__', 'A', weight=1e-9)
    task_graph.add_edge('__src__', 'B', weight=1e-9)
    task_graph.add_edge('B', 'C', weight=0.77)
    task_graph.add_edge('A', '__dst__', weight=1e-9)
    task_graph.add_edge('C', '__dst__', weight=1e-9)

    cpop_schedule = cpop_scheduler.schedule(network, task_graph)
    heft_schedule = heft_scheduler.schedule(network, task_graph)

    cpop_makespan = max([0 if not tasks else tasks[-1].end for tasks in cpop_schedule.values()])
    heft_makespan = max([0 if not tasks else tasks[-1].end for tasks in heft_schedule.values()])

    print(f'CPoP makespan: {cpop_makespan}')
    print(f'HEFT makespan: {heft_makespan}')
    print(f"Makespan Ratio: {heft_makespan/cpop_makespan}")

    # task graph with __src__ and __dst__ nodes removed
    clean_task_graph = task_graph.copy()
    clean_task_graph.remove_node('__src__')
    clean_task_graph.remove_node('__dst__')

    savepath = thisdir / 'heft_vs_cpop'
    savepath.mkdir(exist_ok=True)
    axis = draw_task_graph(
        clean_task_graph, use_latex=True,
        font_size=font_size,
        weight_font_size=font_size
    )
    axis.get_figure().savefig(savepath / 'task_graph.pdf')
    plt.close(axis.get_figure())

    axis = draw_network(
        network, draw_colors=False, use_latex=True,
        font_size=font_size,
        weight_font_size=font_size
    )
    axis.get_figure().savefig(savepath / 'network.pdf')
    plt.close(axis.get_figure())

    max_makespan = max(heft_makespan, cpop_makespan)
    axis = draw_gantt(
        heft_schedule, use_latex=True, xmax=max_makespan,
        font_size=font_size,
        tick_font_size=font_size
    )
    axis.get_figure().savefig(savepath / 'heft_gantt.pdf')
    plt.close(axis.get_figure())

    axis = draw_gantt(
        cpop_schedule, use_latex=True, xmax=max_makespan,
        font_size=font_size,
        tick_font_size=font_size
    )
    axis.get_figure().savefig(savepath / 'cpop_gantt.pdf')
    plt.close(axis.get_figure())


def main():
    cpop_vs_heft()
    heft_vs_cpop()

if __name__ == '__main__':
    main()