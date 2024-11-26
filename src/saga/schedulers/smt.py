from itertools import combinations, product
from typing import Dict, Hashable, List, Optional

import networkx as nx
from pysmt.shortcuts import (GE, LE, And, Div, ExactlyOne, Implies, 
                             Or, Plus, Max, Times, Ite,
                             Iff, Real, Symbol, get_model)
from pysmt.typing import REAL

from saga.scheduler import Task

from ..scheduler import Scheduler, Task


class SMTScheduler(Scheduler):
    """SMT-based scheduler"""
    def __init__(self,
                 epsilon: float = 1e-3,
                 solver_name: Optional[str] = None,
                 mode: str = "makespan") -> None:
        """Initializes the scheduler

        Args:
            epsilon (float, optional): The epsilon value. Defaults to 1e-3.
            solver_name (Optional[str], optional): The name of the solver. Defaults to None.
            mode (str, optional): The mode of the scheduler (either "makespan" or "bottleneck"). Defaults to "makespan".
        """
        super(SMTScheduler, self).__init__()
        self.epsilon = epsilon
        self.solver_name = solver_name
        self.mode = mode

    @classmethod
    def get_assignment_symbols(cls, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[int, Dict[int, Symbol]]:
        """Returns the assignment symbols for the network and task graph

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[int, Dict[int, Symbol]]: The assignment symbols for the network and task graph.
        """
        assignments = {}
        for node_name in network.nodes:
            assignments[node_name] = {}
            for task_name in task_graph.nodes:
                assignments[node_name][task_name] = Symbol(f"Node{node_name}_Task{task_name}", REAL)
        return assignments

    def _schedule(self,
                  network: nx.Graph,
                  task_graph: nx.DiGraph,
                  value: float) -> Optional[Dict[Hashable, List[Task]]]:
        """Returns the schedule of the tasks on the network if one exists within the value

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.
            value (float): The value.

        Returns:
            Optional[Dict[Hashable, List[Task]]]: The schedule of the tasks on the network.
        """
        start_time = self.get_assignment_symbols(network, task_graph)

        constraints = []
        # Each task is assigned to exactly one node
        for task in task_graph.nodes:
            constraints.append(
                ExactlyOne(
                    GE(start_time[node][task], Real(0)) for node in network.nodes
                )
            )

        # add constraints that some tasks cannot be scheduled on some nodes
        schedule_restrictions: Dict[str, List[str]] = task_graph.graph.get('schedule_restrictions', {})
        for task, bad_nodes in schedule_restrictions.items():
            for node in bad_nodes:
                constraints.append(
                    LE(start_time[node][task], Real(-1))
                )

        # Each node can only execute one task at a time
        for node in network.nodes:
            for task1, task2 in combinations(task_graph.nodes, r=2):
                constraints.append(
                    Implies(
                        And(
                            GE(start_time[node][task1], Real(0)),
                            GE(start_time[node][task2], Real(0))
                        ),
                        Or(
                            # task1 ends before task2 starts
                            LE(
                                Plus(start_time[node][task1], Div(Real(task_graph.nodes[task1]['weight']),
                                                                  Real(network.nodes[node]['weight']))),
                                start_time[node][task2]
                            ),
                            # task2 ends before task1 starts
                            LE(
                                Plus(start_time[node][task2], Div(Real(task_graph.nodes[task2]['weight']),
                                                                  Real(network.nodes[node]['weight']))),
                                start_time[node][task1]
                            )
                        )
                    )
                )

        # communication time constraints
        for task1, task2 in task_graph.edges:
            for node1, node2 in product(network.nodes, repeat=2):
                constraints.append(
                    Implies(
                        # task1 and task2 are assigned to node1 and node2 respectively
                        And(
                            GE(start_time[node1][task1], Real(0)),
                            GE(start_time[node2][task2], Real(0))
                        ),
                        # task2 start time is at least task1 end time + communication time
                        LE(
                            Plus(
                                start_time[node1][task1], # start time of task1
                                Div(Real(task_graph.nodes[task1]['weight']),
                                    Real(network.nodes[node1]['weight'])), # execution time of task1
                                Div(Real(task_graph.edges[task1, task2]['weight']),
                                    Real(network.edges[node1, node2]['weight'])) # communication time
                            ),
                            start_time[node2][task2]
                        )
                    )
                )

        
        if self.mode == "makespan": # makespan constraint
            for node, task in product(network.nodes, task_graph.nodes):
                constraints.append(
                    Implies(
                        GE(start_time[node][task], Real(0)),
                        LE(
                            Plus(start_time[node][task], Div(Real(task_graph.nodes[task]['weight']),
                                                             Real(network.nodes[node]['weight']))),
                            Real(value)
                        )
                    )
                )
        elif self.mode == "bottleneck": # bottleneck constraint
            for node in network.nodes:
                # add constraint that the sum of the execution times of all tasks assigned to the node is at most value
                constraints.append(
                    LE(
                        Plus(
                            Ite(
                                GE(start_time[node][task], Real(0)),
                                Div(Real(task_graph.nodes[task]['weight']), Real(network.nodes[node]['weight'])),
                                Real(0)
                            )
                            for task in task_graph.nodes
                        ),
                        Real(value)
                    )
                )
            for src, dst in product(network.nodes, repeat=2):
                # add constraint that the sum of the communication times of all tasks assigned to the node is at most value
                constraints.append(
                    LE(
                        Max(
                            Ite(
                                And( # task1 and task2 are assigned to src and dst respectively
                                    GE(start_time[src][task1], Real(0)),
                                    GE(start_time[dst][task2], Real(0))
                                ),
                                Div(Real(task_graph.edges[task1, task2]['weight']), Real(network.edges[src, dst]['weight'])),
                                Real(0)
                            )
                            for task1, task2 in task_graph.edges
                        ),
                        Real(value)
                    )
                )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
                

        model = get_model(And(constraints), solver_name=self.solver_name)
        schedule: Dict[Hashable, List[Task]] = {}
        if model:
            for node in network.nodes:
                schedule[node] = []
                for task in task_graph.nodes:
                    if model.get_py_value(start_time[node][task]) >= 0:
                        schedule[node].append(
                            Task(
                                node=node,
                                name=task,
                                start=float(model.get_py_value(start_time[node][task])),
                                end=(float(model.get_py_value(start_time[node][task])) +
                                     task_graph.nodes[task]['weight'] / network.nodes[node]['weight'])
                            )
                        )
            # sort tasks by start time
            for _, tasks in schedule.items():
                tasks.sort(key=lambda task: task.start)
            return schedule
        else:
            return None


    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Returns an epsilon-optimal schedule of the tasks on the network.

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: The schedule of the tasks on the network.
        """
        # binary search for the makespan
        lower_bound = 0
        # upper_bound is if we execute all tasks on the fastest node
        # This doesn't work if there are schedule restrictions, so this is just a starting point
        fastest_node = max(network.nodes, key=lambda node: network.nodes[node]['weight'])

        upper_bound = 1
        # for task in task_graph.nodes:
        #     exec_time = task_graph.nodes[task]['weight'] / network.nodes[fastest_node]['weight']
        #     comm_time = max([
        #         task_graph.edges[pred_task, task]['weight'] / network.edges[pred_node, fastest_node]['weight']
        #         for pred_task, pred_node in product(task_graph.predecessors(task), network.nodes)
        #     ], default=0)
        #     upper_bound += exec_time + comm_time

        schedule = None
        while schedule is None: 
            # if the upper bound is too low (because of schedule restrictions), double it and try again
            upper_bound *= 2
            schedule = self._schedule(network, task_graph, upper_bound)
        if schedule is None:
            raise ValueError("Error in SMT solver, a schedule should always exist")
        makespan = (lower_bound + upper_bound) / 2
        while upper_bound - lower_bound > self.epsilon:
            _schedule = self._schedule(network, task_graph, makespan)
            if _schedule is not None:
                upper_bound = makespan
                schedule = _schedule
            else:
                lower_bound = makespan
            makespan = (lower_bound + upper_bound) / 2

        return schedule
