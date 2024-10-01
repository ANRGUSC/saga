from typing import Any, Dict, Hashable, List

from matplotlib import pyplot as plt
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import ArbitraryTopological, GreedyInsert, CPoPRanking
from saga.pisa import run_experiments
from saga.pisa.simulated_annealing import SimulatedAnnealing, SimulatedAnnealingIteration
from saga.pisa.changes import TaskGraphChangeDependencyWeight, TaskGraphChangeTaskWeight
from saga.schedulers.parametric import IntialPriority
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
import networkx as nx

import pathlib

thisdir = pathlib.Path(__file__).parent.resolve()

def main():
    scheduler = ParametricScheduler(
        initial_priority=CPoPRanking(),
        insert_task=GreedyInsert(
            append_only=False,
            compare="EST",
            critical_path=True
        )
    )

    scheduler_append_only = ParametricScheduler(
        initial_priority=scheduler.initial_priority,
        insert_task=GreedyInsert(
            append_only=True,
            compare=scheduler.insert_task.compare,
            critical_path=scheduler.insert_task.critical_path
        )
    )

    run_experiments(
        scheduler_pairs=[
            (("Insertion Based", scheduler), ("Append Only", scheduler_append_only)),
        ],
        max_iterations=1000,
        num_tries=10,
        max_temp=10,
        min_temp=0.1,
        cooling_rate=0.99,
        skip_existing=False,
        thisdir=thisdir
    )

def example():
    task_graph = nx.DiGraph()
    task_graph.add_node("A", weight=0.5)
    task_graph.add_node("B", weight=0.5)
    task_graph.add_node("C", weight=0.5)
    task_graph.add_node("D", weight=0.5)
    task_graph.add_node("E", weight=0.5)

    task_graph.add_edge("A", "C", weight=0.5)
    task_graph.add_edge("B", "C", weight=0.5)
    task_graph.add_edge("D", "E", weight=0.5)
    task_graph.add_edge("B", "E", weight=0.5)

    network = nx.Graph()
    network.add_node("N1", weight=0.5)
    network.add_node("N2", weight=1)
    network.add_edge("N1", "N2", weight=1)
    network.add_edge("N1", "N1", weight=1e9)
    network.add_edge("N2", "N2", weight=1e9)

    class PriorityOrder(IntialPriority):
        def __call__(self,
                    network: nx.Graph,
                    task_graph: nx.DiGraph) -> List[Hashable]:
            return ["__src__", "A", "B", "D", "C", "E", "__dst__"]
        
        def serialize(self) -> Dict[str, Any]:
            return {"name": "PriorityOrder"}
        
        @classmethod
        def deserialize(cls, data: Dict[str, Any]) -> "ArbitraryTopological":
            return cls()

    scheduler = ParametricScheduler(
        initial_priority=PriorityOrder(),
        insert_task=GreedyInsert(
            append_only=False,
            compare="EFT",
            critical_path=False
        )
    )
    scheduler_append_only = ParametricScheduler(
        initial_priority=scheduler.initial_priority,
        insert_task=GreedyInsert(
            append_only=True,
            compare=scheduler.insert_task.compare,
            critical_path=scheduler.insert_task.critical_path
        )
    )

    sa = SimulatedAnnealing(
        task_graph=task_graph,
        network=network,
        scheduler=scheduler,
        base_scheduler=scheduler_append_only,
        max_iterations=1000,
        max_temp=10,
        min_temp=0.1,
        cooling_rate=0.999,
        change_types=[
            TaskGraphChangeDependencyWeight,
            TaskGraphChangeTaskWeight,
            # NetworkChangeEdgeWeight,
            # NetworkChangeNodeWeight
        ]
    )

    result: SimulatedAnnealingIteration = sa.run()
    
    ax = draw_gantt(result.best_schedule)
    plt.tight_layout()
    ax.figure.savefig(thisdir / "task_types_schedule.png")

    ax = draw_network(result.best_network)
    plt.tight_layout()
    ax.figure.savefig(thisdir / "task_types_network.png")

    ax = draw_task_graph(result.best_task_graph)
    plt.tight_layout()
    ax.figure.savefig(thisdir / "task_types_task_graph.png")

def bad_example():
    task_graph = nx.DiGraph()
    task_graph.add_node("__src__", weight=1e-9)
    task_graph.add_node("A", weight=2)
    task_graph.add_node("B", weight=1)
    task_graph.add_node("C", weight=2)
    task_graph.add_node("D", weight=1)
    task_graph.add_node("E", weight=1)
    task_graph.add_node("__dst__", weight=1e-9)

    task_graph.add_edge("__src__", "A", weight=1e-9)
    task_graph.add_edge("__src__", "B", weight=1e-9)
    task_graph.add_edge("__src__", "D", weight=1e-9)
    task_graph.add_edge("A", "C", weight=1)
    task_graph.add_edge("B", "C", weight=1)
    task_graph.add_edge("D", "E", weight=1)
    task_graph.add_edge("B", "E", weight=1)
    task_graph.add_edge("C", "__dst__", weight=1e-9)
    task_graph.add_edge("E", "__dst__", weight=1e-9)

    network = nx.Graph()
    network.add_node("N1", weight=1)
    network.add_node("N2", weight=2)
    network.add_edge("N1", "N2", weight=1)
    network.add_edge("N1", "N1", weight=1e9)
    network.add_edge("N2", "N2", weight=1e9)

    class PriorityOrder(IntialPriority):
        def __call__(self,
                    network: nx.Graph,
                    task_graph: nx.DiGraph) -> List[Hashable]:
            return ["__src__", "A", "B", "C", "D", "E", "__dst__"]
        
        def serialize(self) -> Dict[str, Any]:
            return {"name": "PriorityOrder"}
        
        @classmethod
        def deserialize(cls, data: Dict[str, Any]) -> "ArbitraryTopological":
            return cls()

    scheduler = ParametricScheduler(
        initial_priority=PriorityOrder(),
        insert_task=GreedyInsert(
            append_only=False,
            compare="EFT",
            critical_path=False
        )
    )
    scheduler_append_only = ParametricScheduler(
        initial_priority=scheduler.initial_priority,
        insert_task=GreedyInsert(
            append_only=True,
            compare=scheduler.insert_task.compare,
            critical_path=scheduler.insert_task.critical_path
        )
    )

    schedule = scheduler.schedule(network, task_graph)
    schedule_append_only = scheduler_append_only.schedule(network, task_graph)

    ax = draw_gantt(schedule)
    plt.tight_layout()
    ax.figure.savefig(thisdir / "bad_example_schedule.png")

    ax = draw_gantt(schedule_append_only)
    plt.tight_layout()
    ax.figure.savefig(thisdir / "bad_example_schedule_append_only.png")

    ax = draw_network(network)
    plt.tight_layout()
    ax.figure.savefig(thisdir / "bad_example_network.png")

if __name__ == "__main__":
    # main()
    # example()
    bad_example()
