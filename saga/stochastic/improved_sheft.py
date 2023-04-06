import logging
import networkx as nx
from typing import Dict, Hashable, List, Tuple

from saga.utils.random_variable import RandomVariable
from .stoch_heft import stoch_heft_rank_sort, StochHeftScheduler
from ..common.heft import HeftScheduler
from ..base import Scheduler, Task

class ImprovedSheftScheduler(Scheduler):
    def __init__(self):
        super().__init__()   

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        runtimes, commtimes = StochHeftScheduler.get_runtimes(network, task_graph)
        schedule_order = stoch_heft_rank_sort(network, task_graph, runtimes, commtimes) 
        scheduler = HeftScheduler()
        schedule = scheduler._schedule(
            network, task_graph, 
            runtimes, commtimes,
            schedule_order
        )
        return schedule