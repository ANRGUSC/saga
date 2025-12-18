from itertools import combinations, product
from typing import Dict, Optional, Any

from pysmt.shortcuts import (
    GE,
    LE,
    And,
    Div,
    ExactlyOne,
    Implies,
    Or,
    Plus,
    Real,
    Symbol,
    get_model,
)
from pysmt.typing import REAL

from saga import Network, Schedule, Scheduler, ScheduledTask, TaskGraph


class SMTScheduler(Scheduler):
    """SMT-based scheduler"""

    def __init__(
        self, epsilon: float = 1e-3, solver_name: Optional[str] = None
    ) -> None:
        """Initializes the scheduler

        Args:
            epsilon (float, optional): The epsilon value. Defaults to 1e-3.
            solver_name (Optional[str], optional): The name of the solver. Defaults to None.
        """
        super(SMTScheduler, self).__init__()
        self.epsilon = epsilon
        self.solver_name = solver_name

    @classmethod
    def get_assignment_symbols(
        cls, network: Network, task_graph: TaskGraph
    ) -> Dict[str, Dict[str, Any]]:
        """Returns the assignment symbols for the network and task graph

        Args:
            network (Network): The network.
            task_graph (TaskGraph): The task graph.

        Returns:
            Dict[str, Dict[str, Symbol]]: The assignment symbols for the network and task graph.
        """
        assignments: Dict[str, Dict[str, Any]] = {}
        for node in network.nodes:
            assignments[node.name] = {}
            for task in task_graph.tasks:
                assignments[node.name][task.name] = Symbol(
                    f"Node{node.name}_Task{task.name}", REAL
                )
        return assignments

    def _schedule(
        self, network: Network, task_graph: TaskGraph, makespan: float
    ) -> Optional[Schedule]:
        """Returns the schedule of the tasks on the network if one exists within the makespan

        Args:
            network (Network): The network.
            task_graph (TaskGraph): The task graph.
            makespan (float): The makespan.

        Returns:
            Optional[Schedule]: The schedule of the tasks on the network.
        """
        start_time = self.get_assignment_symbols(network, task_graph)

        node_names = [node.name for node in network.nodes]
        task_names = [task.name for task in task_graph.tasks]

        constraints = []
        # Each task is assigned to exactly one node
        for task_name in task_names:
            constraints.append(
                ExactlyOne(
                    GE(start_time[node_name][task_name], Real(0))
                    for node_name in node_names
                )
            )

        # Each node can only execute one task at a time
        for node_name in node_names:
            node = network.get_node(node_name)
            for task1_name, task2_name in combinations(task_names, r=2):
                task1 = task_graph.get_task(task1_name)
                task2 = task_graph.get_task(task2_name)
                constraints.append(
                    Implies(
                        And(
                            GE(start_time[node_name][task1_name], Real(0)),
                            GE(start_time[node_name][task2_name], Real(0)),
                        ),
                        Or(
                            # task1 ends before task2 starts
                            LE(
                                Plus(
                                    start_time[node_name][task1_name],
                                    Div(Real(task1.cost), Real(node.speed)),
                                ),
                                start_time[node_name][task2_name],
                            ),
                            # task2 ends before task1 starts
                            LE(
                                Plus(
                                    start_time[node_name][task2_name],
                                    Div(Real(task2.cost), Real(node.speed)),
                                ),
                                start_time[node_name][task1_name],
                            ),
                        ),
                    )
                )

        # communication time constraints
        for dep in task_graph.dependencies:
            task1 = task_graph.get_task(dep.source)
            for node1_name, node2_name in product(node_names, repeat=2):
                node1 = network.get_node(node1_name)
                edge = network.get_edge(node1_name, node2_name)
                constraints.append(
                    Implies(
                        # task1 and task2 are assigned to node1 and node2 respectively
                        And(
                            GE(start_time[node1_name][dep.source], Real(0)),
                            GE(start_time[node2_name][dep.target], Real(0)),
                        ),
                        # task2 start time is at least task1 end time + communication time
                        LE(
                            Plus(
                                start_time[node1_name][
                                    dep.source
                                ],  # start time of task1
                                Div(
                                    Real(task1.cost), Real(node1.speed)
                                ),  # execution time of task1
                                Div(
                                    Real(dep.size), Real(edge.speed)
                                ),  # communication time
                            ),
                            start_time[node2_name][dep.target],
                        ),
                    )
                )

        # makespan constraint
        for node_name, task_name in product(node_names, task_names):
            task = task_graph.get_task(task_name)
            node = network.get_node(node_name)
            constraints.append(
                Implies(
                    GE(start_time[node_name][task_name], Real(0)),
                    LE(
                        Plus(
                            start_time[node_name][task_name],
                            Div(Real(task.cost), Real(node.speed)),
                        ),
                        Real(makespan),
                    ),
                )
            )

        model = get_model(And(constraints), solver_name=self.solver_name)
        if model:
            comp_schedule = Schedule(task_graph, network)
            for node_name in node_names:
                node = network.get_node(node_name)
                for task_name in task_names:
                    task = task_graph.get_task(task_name)
                    if model.get_py_value(start_time[node_name][task_name]) >= 0:
                        new_task = ScheduledTask(
                            node=node_name,
                            name=task_name,
                            start=float(
                                model.get_py_value(start_time[node_name][task_name])
                            ),
                            end=(
                                float(
                                    model.get_py_value(start_time[node_name][task_name])
                                )
                                + task.cost / node.speed
                            ),
                        )
                        comp_schedule.add_task(new_task)
            return comp_schedule
        else:
            return None

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Returns an epsilon-optimal schedule of the tasks on the network.

        Args:
            network (Network): The network.
            task_graph (TaskGraph): The task graph.
            schedule (Optional[Schedule]): Ignored (SMT scheduler doesn't support incremental scheduling).
            min_start_time (float): Ignored (SMT scheduler doesn't support min_start_time).

        Returns:
            Schedule: The schedule of the tasks on the network.
        """
        # binary search for the makespan
        lower_bound = 0.0
        # upper_bound is if we execute all tasks on the fastest node
        fastest_node = max(network.nodes, key=lambda node: node.speed)

        upper_bound = 0.0
        for task in task_graph.tasks:
            exec_time = task.cost / fastest_node.speed
            comm_time = max(
                [
                    dep.size / network.get_edge(pred_node.name, fastest_node.name).speed
                    for dep in task_graph.in_edges(task.name)
                    for pred_node in network.nodes
                ],
                default=0,
            )
            upper_bound += exec_time + comm_time

        result_schedule = self._schedule(network, task_graph, upper_bound)
        if result_schedule is None:
            raise ValueError("Error in SMT solver, a schedule should always exist")
        makespan = (lower_bound + upper_bound) / 2
        while upper_bound - lower_bound > self.epsilon:
            _schedule = self._schedule(network, task_graph, makespan)
            if _schedule is not None:
                upper_bound = makespan
                result_schedule = _schedule
            else:
                lower_bound = makespan
            makespan = (lower_bound + upper_bound) / 2

        return result_schedule
