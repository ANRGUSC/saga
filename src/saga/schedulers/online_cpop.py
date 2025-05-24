from functools import partial
from typing import Dict, Hashable, List, Tuple
from copy import deepcopy
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from saga.utils.draw import gradient_heatmap
from multiprocessing import Pool, Value, Lock, cpu_count

from saga.schedulers import CpopScheduler
from saga.scheduler import Scheduler, Task
from saga.utils.online_tools import schedule_estimate_to_actual, get_offline_instance

class OnlineCpopScheduler(Scheduler):
    def __init__(self):
        self.cpop_scheduler = CpopScheduler()

    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
