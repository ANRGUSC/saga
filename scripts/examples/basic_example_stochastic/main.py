from functools import lru_cache
import logging
import pathlib
from typing import Tuple
import networkx as nx

from saga.stochastic import StochasticNetwork, StochasticTaskGraph, StochasticSchedule
from saga.schedulers import HeftScheduler
from saga.schedulers.stochastic.determinizer import Determinizer
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from saga.utils.random_variable import RandomVariable, UniformRandomVariable

logging.basicConfig(level=logging.INFO)

thisdir = pathlib.Path(__file__).parent.absolute()
savedir = thisdir / 'outputs'
savedir.mkdir(exist_ok=True)

@lru_cache(maxsize=None)
def get_instance() -> Tuple[StochasticNetwork, StochasticTaskGraph]:
    network = nx.Graph()
    network.add_node("v_1", weight=UniformRandomVariable(1, 3))
    network.add_node("v_2", weight=UniformRandomVariable(2, 4))
    network.add_node("v_3", weight=UniformRandomVariable(1, 2))
    network.add_edge("v_1", "v_2", weight=UniformRandomVariable(1, 3))
    network.add_edge("v_1", "v_3", weight=UniformRandomVariable(2, 5))
    network.add_edge("v_2", "v_3", weight=UniformRandomVariable(1, 4))

    task_graph = nx.DiGraph()
    task_graph.add_node('t_1', weight=UniformRandomVariable(2, 5))
    task_graph.add_node('t_2', weight=UniformRandomVariable(3, 6))
    task_graph.add_node('t_3', weight=UniformRandomVariable(2, 4))
    task_graph.add_node('t_4', weight=UniformRandomVariable(6, 10))

    task_graph.add_edge('t_1', 't_2', weight=UniformRandomVariable(1, 3))
    task_graph.add_edge('t_1', 't_3', weight=UniformRandomVariable(2, 4))
    task_graph.add_edge('t_2', 't_4', weight=UniformRandomVariable(1, 2))
    task_graph.add_edge('t_3', 't_4', weight=UniformRandomVariable(3, 5))

    nw = StochasticNetwork.from_nx(network)
    tg = StochasticTaskGraph.from_nx(task_graph)
    return nw, tg

def main():
    scheduler = Determinizer(
        scheduler=HeftScheduler(),
        determinize=lambda rv: rv.mean()
    )
    network, task_graph = get_instance()
    schedule = scheduler.schedule(network, task_graph)

    print(f"Makespan: {schedule.makespan}")
    
if __name__ == '__main__':
    main()
