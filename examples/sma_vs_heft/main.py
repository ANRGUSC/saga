import networkx as nx
from src.saga.schedulers.heft import HeftScheduler
from src.saga.schedulers.sma import SimulatedAnnealingScheduler
from src.saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from scripts.experiments.benchmarking.prepare import scale_ccr
from src.saga.utils.random_graphs import (
    add_random_weights,
    get_branching_dag,
    get_chain_dag,
    get_diamond_dag,
    get_fork_dag,
    get_network,
)
import pathlib
import matplotlib.pyplot as plt
from typing import Dict, List
import random

thisdir = pathlib.Path(__file__).parent.absolute()

def sma_vs_heft(network: nx.Graph, task_graph: nx.DiGraph):
  sma_scheduler = SimulatedAnnealingScheduler(        
        initial_temperature = 100,
        cooling_rate= 0.8,
        stopping_temperature= 0.0001,
        num_iterations = 750)
  heft_scheduler = HeftScheduler()

  heft_schedule = heft_scheduler.schedule(network, task_graph)
  sma_schedule = sma_scheduler.schedule(network, task_graph)

  heft_makespan = max([0 if not tasks else tasks[-1].end for tasks in heft_schedule.values()])
  sma_makespan = max([0 if not tasks else tasks[-1].end for tasks in sma_schedule.values()])

  print(f'HEFT makespan: {heft_makespan}')
  print(f'Sma makespan: {sma_makespan}')
  print(f"Makespan Ratio: {sma_makespan/heft_makespan}")

  
  clean_task_graph = task_graph.copy()

  savepath = thisdir / 'sma_vs_heft'
  savepath.mkdir(exist_ok=True)
  axis = draw_task_graph(clean_task_graph, use_latex=False)
  axis.get_figure().savefig(savepath / 'task_graph.png')
  plt.close(axis.get_figure())
  
  axis = draw_network(network, draw_colors=False, use_latex=False)
  axis.get_figure().savefig(savepath / 'network.png')
  
  max_makespan = max(heft_makespan, sma_makespan)
  axis = draw_gantt(sma_schedule, use_latex=False, xmax=max_makespan)
  axis.get_figure().savefig(savepath / 'sma_gantt.png')
  plt.close(axis.get_figure())

  axis = draw_gantt(heft_schedule, use_latex=False, xmax=max_makespan)
  axis.get_figure().savefig(savepath / 'heft_gantt.png')
  plt.close(axis.get_figure())

  return heft_makespan, sma_makespan


def random_network():
  num_nodes = random.randint(1, 3)
  network = add_random_weights(get_network(num_nodes))
  print(f'num_nodes: {num_nodes}')
  return network

def random_task_graph():
  num_tasks = random.randint(3, 5)
  task_graph = add_random_weights(get_branching_dag(3, 2))
  print(f'num_tasks: {num_tasks}')
  return task_graph

def grid_search(ccrs : List[float]):
  # Define network
  network = nx.Graph()
  network.add_node(1, weight=0.99)
  network.add_node(2, weight=0.31)
  network.add_node(3, weight=0.98)
  network.add_edge(1, 2, weight=0.88)
  network.add_edge(1, 3, weight=0.61)
  network.add_edge(2, 3, weight=0.72)
  network.add_edge(1, 1, weight=1e9)
  network.add_edge(2, 2, weight=1e9)
  network.add_edge(3, 3, weight=1e9)

  # Define task_graph
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
  heft_makespans = []
  sma_makespans = []
  for ccr in ccrs:
    scale_ccr(task_graph, network, ccr)
    heft_makespan, sma_makespan = sma_vs_heft(network, task_graph)
    heft_makespans.append((ccr, heft_makespan))
    sma_makespans.append((ccr, sma_makespan))

  plt.figure(figsize=(10, 6))
  plt.plot([heft_makespans, sma_makespans], tick_labels=['HEFT', 'SMA'])
  plt.ylabel('Makespan')
  plt.title('Comparison of Makespans: HEFT vs SMA')
  
  # Save the plot
  plt.savefig("sma_vs_heft_makespans.png")
  plt.close()
  print(f'Heft Makespans with CCRs {heft_makespans}')
  print(f'SMA makespans with CCRs {sma_makespans}')



     
    
def get_makespans():
    heft_makespans = []
    sma_makespans = []
    
    # Run the simulations and collect makespans for both HEFT and SMA
    for i in range(10):
        heft_makespan, sma_makespan = sma_vs_heft(random_network(), random_task_graph())
        heft_makespans.append(heft_makespan)
        sma_makespans.append(sma_makespan)
    
    # Save the makespans to a file
    with open('makespans.txt', "w+") as f:
        f.writelines([f"{sma} {heft}\n" for (heft, sma) in zip(heft_makespans, sma_makespans)])
    
    # Create a boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot([heft_makespans, sma_makespans], tick_labels=['HEFT', 'SMA'])
    plt.ylabel('Makespan')
    plt.title('Comparison of Makespans: HEFT vs SMA')
    
    # Save the plot
    plt.savefig("sma_vs_heft_makespans.png")
    plt.close()

def main():
  ccrs = [0.2, 0.5, 1, 2, 5]
  grid_search(ccrs)
  get_makespans()
  

main()