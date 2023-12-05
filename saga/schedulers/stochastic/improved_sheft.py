import logging
from typing import Dict, Hashable, List, Tuple

import networkx as nx

from saga.utils.random_variable import RandomVariable

from ...scheduler import Scheduler, Task
from ..heft import HeftScheduler
from .stoch_heft import StochHeftScheduler, stoch_heft_rank_sort


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