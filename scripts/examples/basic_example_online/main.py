import logging
import pathlib
from functools import lru_cache
from typing import Tuple

import networkx as nx

from saga import Network, Schedule, TaskGraph
from saga.schedulers.parametric.online import OnlineParametricScheduler, schedulers
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.stochastic.estimate_stochastic_scheduler import EstimateStochasticScheduler
from saga.stochastic import StochasticNetwork, StochasticTaskGraph
from saga.utils.random_variable import UniformRandomVariable
from saga.schedulers.parametric.components import (
    GreedyInsert,
    GreedyInsertCompareFuncs,
    UpwardRanking,
)
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph


logging.basicConfig(level=logging.INFO)

thisdir = pathlib.Path(__file__).parent.absolute()
savedir = thisdir / "outputs"
savedir.mkdir(exist_ok=True)



@lru_cache(maxsize=None)
def get_instance() -> Tuple[StochasticNetwork, StochasticTaskGraph]:
    network = nx.Graph()
    network.add_node("v_1", weight=UniformRandomVariable(3, 4))
    network.add_node("v_2", weight=UniformRandomVariable(3, 4))
    network.add_node("v_3", weight=UniformRandomVariable(3, 4))
    network.add_edge("v_1", "v_2", weight=UniformRandomVariable(10, 15))
    network.add_edge("v_1", "v_3", weight=UniformRandomVariable(10,15))
    network.add_edge("v_2", "v_3", weight=UniformRandomVariable(10, 15))

    task_graph = nx.DiGraph()
    task_graph.add_node("t_1", weight=UniformRandomVariable(2, 5))
    task_graph.add_node("t_2", weight=UniformRandomVariable(3, 6))
    task_graph.add_node("t_3", weight=UniformRandomVariable(2, 4))
    task_graph.add_node("t_4", weight=UniformRandomVariable(6, 10))

    # task_graph.add_node("t_5", weight=UniformRandomVariable(2, 5))
    # task_graph.add_node("t_6", weight=UniformRandomVariable(3, 6))
    # task_graph.add_node("t_7", weight=UniformRandomVariable(2, 4))
    # task_graph.add_node("t_8", weight=UniformRandomVariable(6, 10))

    task_graph.add_edge("t_1", "t_2", weight=UniformRandomVariable(1, 3))
    task_graph.add_edge("t_1", "t_3", weight=UniformRandomVariable(2, 4))
    task_graph.add_edge("t_2", "t_4", weight=UniformRandomVariable(1, 2))
    task_graph.add_edge("t_3", "t_4", weight=UniformRandomVariable(3, 5))

    # task_graph.add_edge("t_1", "t_5", weight=UniformRandomVariable(1, 3))
    # task_graph.add_edge("t_1", "t_6", weight=UniformRandomVariable(2, 4))
    # task_graph.add_edge("t_2", "t_7", weight=UniformRandomVariable(1, 2))
    # task_graph.add_edge("t_3", "t_8", weight=UniformRandomVariable(3, 5))

    nw = StochasticNetwork.from_nx(network)
    tg = StochasticTaskGraph.from_nx(task_graph)
    return nw, tg

def parametric_schedule():
    network, task_graph = get_instance()
    internal_scheduler = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(
            append_only=False, compare=GreedyInsertCompareFuncs.EFT, critical_path=False
        ),
    )
    scheduler = OnlineParametricScheduler(scheduler=internal_scheduler, estimate=lambda rv: rv.mean())
    schedule = scheduler.schedule(network, task_graph)
    return schedule

def draw_instance(network: Network, task_graph: TaskGraph):
    logging.basicConfig(level=logging.INFO)
    ax = draw_task_graph(task_graph.graph, use_latex=False)
    fig = ax.get_figure()
    try:
        fig.savefig(str(savedir / "task_graph.png"))  # type: ignore
    except Exception as e:
        logging.error(f"Failed to save task graph figure: {e}")

    ax = draw_network(network.graph, draw_colors=False, use_latex=False)
    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(str(savedir / "network.png"))  # type: ignore


def draw_schedule(schedule: Schedule, name: str, xmax: float | None = None):
    ax = draw_gantt(schedule.mapping, use_latex=False, xmax=xmax)
    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(str(savedir / f"{name}.png"))  # type: ignore

def main():
    
    parametric_sched = parametric_schedule()
    print(
        f"Makespan {parametric_sched.makespan} "
    )
    draw_schedule(parametric_sched, "online_gantt_scaled", xmax=parametric_sched.makespan)
    #max_makespan = max(sched.makespan for sched in [heft_sched, parametric_sched])


if __name__ == "__main__":
    main()
