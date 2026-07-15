import logging
from functools import lru_cache
from typing import Tuple

import networkx as nx

from saga.schedulers import HeftScheduler
from saga.schedulers.stochastic.determinizer import Determinizer
from saga.stochastic import StochasticNetwork, StochasticTaskGraph
from saga.utils.random_variable import UniformRandomVariable

logging.basicConfig(level=logging.INFO)


@lru_cache(maxsize=None)
def get_instance() -> Tuple[StochasticNetwork, StochasticTaskGraph]:
    '''
    network = nx.Graph()
    network.add_node("v_1", weight=UniformRandomVariable(1, 3))
    network.add_node("v_2", weight=UniformRandomVariable(2, 4))
    network.add_node("v_3", weight=UniformRandomVariable(1, 2))
    network.add_edge("v_1", "v_2", weight=UniformRandomVariable(1, 3))
    network.add_edge("v_1", "v_3", weight=UniformRandomVariable(2, 5))
    network.add_edge("v_2", "v_3", weight=UniformRandomVariable(1, 4))

    task_graph = nx.DiGraph()
    task_graph.add_node("t_1", weight=UniformRandomVariable(2, 5))
    task_graph.add_node("t_2", weight=UniformRandomVariable(3, 6))
    task_graph.add_node("t_3", weight=UniformRandomVariable(2, 4))
    task_graph.add_node("t_4", weight=UniformRandomVariable(6, 10))

    task_graph.add_edge("t_1", "t_2", weight=UniformRandomVariable(1, 3))
    task_graph.add_edge("t_1", "t_3", weight=UniformRandomVariable(2, 4))
    task_graph.add_edge("t_2", "t_4", weight=UniformRandomVariable(1, 2))
    task_graph.add_edge("t_3", "t_4", weight=UniformRandomVariable(3, 5))
    
    nw = StochasticNetwork.from_nx(network)
    tg = StochasticTaskGraph.from_nx(task_graph)
    '''
    #New way
    nw = StochasticNetwork.create(
        nodes=[
            ("v_1", UniformRandomVariable(1, 3)),
            ("v_2", UniformRandomVariable(2, 4)),
            ("v_3", UniformRandomVariable(1, 2)),
        ],
        edges=[
            ("v_1", "v_2", UniformRandomVariable(1, 3)),
            ("v_1", "v_3", UniformRandomVariable(2, 5)),
            ("v_2", "v_3", UniformRandomVariable(1, 4)),
        ],
    )
    tg = StochasticTaskGraph.create(
        tasks=[
            ("t_1", UniformRandomVariable(2, 5)),
            ("t_2", UniformRandomVariable(3, 6)),
            ("t_3", UniformRandomVariable(2, 4)),
            ("t_4", UniformRandomVariable(6, 10)),
        ],
        dependencies=[
            ("t_1", "t_2", UniformRandomVariable(1, 3)),
            ("t_1", "t_3", UniformRandomVariable(2, 4)),
            ("t_2", "t_4", UniformRandomVariable(1, 2)),
            ("t_3", "t_4", UniformRandomVariable(3, 5)),
        ],
    )  

    return nw, tg


def main():
    scheduler = Determinizer(
        scheduler=HeftScheduler(), determinize=lambda rv: rv.mean()
    )
    network, task_graph = get_instance()
    schedule = scheduler.schedule(network, task_graph)
    print(schedule.mapping)
    print(f"Makespan: {schedule.makespan}")


if __name__ == "__main__":
    main()
