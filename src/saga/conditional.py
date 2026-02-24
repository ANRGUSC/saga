from abc import ABC, abstractmethod
import bisect
import logging
import math
from functools import cached_property
from itertools import product
from queue import PriorityQueue
from typing import Dict, FrozenSet, Generator, Iterable, List, Optional, Set, Tuple
from statistics import mean

import networkx as nx
from pydantic import BaseModel, Field, PrivateAttr

from saga.stochastic import StochasticTaskGraph, StochasticTaskGraphNode
from saga.utils.random_variable import RandomVariable, DEFAULT_NUM_SAMPLES
from saga import (
    Schedule,
    ScheduledTask,
    TaskGraph,
    Network,
    NetworkNode,
    NetworkEdge,
    TaskGraphNode,
    TaskGraphEdge,
)

#Brainstormed steps
#ConditionalTaskGraphNode can take normal nodes that are not conditional
#implement ConditionalTaskgraph, all nodes are conditionalTaskGraphNodes, track conditional groups
#Implement ConditionalScheduler (wraps any shceduler , access internal ranking (heft rank, ect.) allows overlap)- #Add conditionalTask and conditionalSchedyule classes for schedule output
#Add get_random_conditional_taskgraph() using existing random_graphs.py functions
#test


class ConditionalTaskGraphNode(StochasticTaskGraphNode):
    """Represents a task that can have conditional branches.
    
        What we want to implement, 
        Normal Node (no branch)
        node_a = ConditionalTaskGraphNode(name="A", cost=5.0)
        #not conditional

        Conditional Branch
            node_b = ConditionalTaskGraphNode(
            name="B",
            conditional_branches=[
                (graph1, 0.9),  # 90% chance
                (graph2, 0.1),  # 10% chance
                ]
            )
        #conditional

        Methodology:
        -When someone creates a ConditionalTaskGraphNode

        -Did they provide branches 
            If yes: calculate cost from branch costs 
                    Check probabilities add up to one
            
            If No: They must provide cost directly
                    Set branches to empty list

        cost = weight of nodes
    """
    #@property
    #def is_conditional(self) -> bool:
    #    return len(self.conditional_branches) > 0

    name: str
    conditional_branches: List[Tuple[StochasticTaskGraph, float]] = Field(
        default_factory=list,
        description="List of (task, probability) pairs representing conditional branches.",
    )

    def __init__(self, name: str, cost = None, conditional_branches = None, **data):
        
        #case 1
        if conditional_branches and len(conditional_branches) > 0 :
            cost_by_branch = [
                sum(task.cost for task in branch.tasks) for branch, _ in conditional_branches
            ]

            cost = RandomVariable(samples=[mean(cost_by_branch)])
            
            total_prob = sum(prob for _, prob in conditional_branches)                
            if not math.isclose(total_prob, 1):
                raise ValueError("Probabilities don't add to 1")
                
        #case 2
        else:
            if cost is None:
                raise ValueError("No cost given")
            
            if isinstance(cost, (int, float)):
                cost = RandomVariable(samples=[float(cost)])

            conditional_branches = []

        super().__init__(name=name, cost=cost, **data)
        self.conditional_branches = conditional_branches 
                
        



class ConditionalTaskGraph(StochasticTaskGraph):
    """A TaskGraph that supports conditional branches.
        -Stores only ConditionalTaskGraphNodes
        -Reuses StochasticTaskGraph behavior for DAG/dependencies
        -Computes conditional_groups for sibling branch roots

    """

    #TODO, take in COnditionalTaskGraphNode
    #TODO, firgure out if conditional if so then mark as conditional so easier to group 
    #TODO, use dependencies to build taskgraph

    def __init__(
        self,
        nodes: List[StochasticTaskGraphNode | ConditionalTaskGraphNode],
        edges: List[TaskGraphEdge],
        condition_edges: List[TaskGraphEdge],
    ):
        pass # TODO: What to do...

class ConditionalSchedule():
    """
    -A schedule that has conditional tasks
    -wrapper similar to stochastic scheduler that takes heuristic like heft and a 
    conditionalTaskGraph and NetworkNode and creates a overlapping schedule output like 
    @\\wsl.localhost\Ubuntu\home\gtwiggho\saga_developer\scripts\examples\conditional\overlapping_scheduler.py 
    
    """