"""Conditional scheduling experiment.

Compares branch makespans from overlapping CTG scheduling vs scheduling
each branch as a standalone task graph. Outputs visualisations and a CSV
of results for later analysis.
 
Workflow:
  for each heuristic
    for each network
      for each CTG
        - schedule CTG with overlapping conditional logic
        - extract per-branch schedules (recalculated to remove gaps)
        - schedule each branch standalone (plain TaskGraph, no conditionals)
        - log makespans to CSV
        - save visualisations
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from saga import Network, Scheduler
from saga.schedulers import HeftScheduler
from saga.utils.draw import draw_gantt, draw_task_graph, draw_mutual_exclusion_graph
from saga.utils.random_graphs import get_random_conditional_branching_dag
from saga.conditional import (
    ConditionalTaskGraph,
    extract_branches_with_recalculation,
    schedule_branch_standalone,
)

#Configuration -----------
    #Dictionary of Heuristics
    #Dictionary of Networks
    #Dictionary of CTGs

#Seed for randomisation of different CTGs?
#number of repeats
#random seed

#helper functions

#main experiment loop
#for each heuristic test each network and each CTG
#schedule the CTG first then extract/recalculate branches then schedule each branch standalone
#save graphs and log comparison data row by row
                #output folder saves all main graphs for this experiment
                #for each branch save branch taskgraph and both schedule versions
                #simple output just gives quick side by side gantt comparison
                #save one csv row for this branch comparison

#run experiment then save all collected comparison rows to csv
