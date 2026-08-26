"""Classic structured task graphs: Gaussian elimination and FFT.

Generates the two structured DAG families used to evaluate list schedulers in
the heterogeneous-scheduling literature -- Gaussian elimination and FFT, with
the exact topologies specified in the HEFT/CPoP paper (Topcuoglu, Hariri and
Wu 2002, doi:10.1109/71.993206, Figs. 8 and 10). The same families appear in
the evaluation of PEFT (Arabnejad and Barbosa 2014, doi:10.1109/TPDS.2013.57).
Only the topologies follow the paper: task costs and dependency sizes are
drawn from SAGA's random-weight utilities rather than the paper's
cost-assignment protocol (the paper, for example, assigns level-uniform
costs to FFT tasks).

The script verifies each generated structure against the paper's closed-form
task counts, schedules it on a random heterogeneous network with several SAGA
schedulers, and draws the instances and Gantt charts.
"""

import logging
import math
import pathlib
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from saga import Network, Schedule, ScheduledTask, TaskGraph
from saga.schedulers import (
    CpopScheduler,
    ETFScheduler,
    HeftScheduler,
    MaxMinScheduler,
    MinMinScheduler,
    PEFTScheduler,
    SufferageScheduler,
)
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from saga.utils.random_graphs import add_random_weights, get_network
from saga.utils.random_variable import RandomVariable, UniformRandomVariable

logging.basicConfig(level=logging.INFO)
# The drawings use node names ("0".."3") as categorical axis values, which
# makes matplotlib's category module log one INFO line per plotted artist.
logging.getLogger("matplotlib").setLevel(logging.WARNING)

thisdir = pathlib.Path(__file__).parent.absolute()
savedir = thisdir / "outputs"
savedir.mkdir(exist_ok=True)


def gaussian_elimination_dag(
    matrix_size: int,
    weight_distribution: Optional[RandomVariable] = None,
) -> TaskGraph:
    """Gaussian elimination task graph for an m x m matrix.

    Structure from the HEFT/CPoP paper (Topcuoglu et al. 2002, Fig. 8):
    elimination step k (k = 1..m-1) has one pivot task T[k,k] and update tasks T[k,j] for
    j = k+1..m. The pivot feeds every update of its step, the first update of
    a step feeds the next step's pivot, and every other update feeds the same
    column's update in the next step. Total tasks: (m^2 + m - 2) / 2.

    Args:
        matrix_size (int): The matrix dimension m (must be at least 2).
        weight_distribution (Optional[RandomVariable]): Distribution for task
            costs and dependency sizes. Defaults to the library default.

    Returns:
        TaskGraph: The Gaussian elimination task graph.

    Raises:
        ValueError: If matrix_size is less than 2.
    """
    m = matrix_size
    if m < 2:
        raise ValueError(f"matrix_size must be at least 2, got {m}")

    def name(k: int, j: int) -> str:
        return f"T[{k},{j}]"

    dag = nx.DiGraph()
    for k in range(1, m):
        dag.add_node(name(k, k))
        for j in range(k + 1, m + 1):
            dag.add_node(name(k, j))
            dag.add_edge(name(k, k), name(k, j))
    for k in range(1, m - 1):
        dag.add_edge(name(k, k + 1), name(k + 1, k + 1))
        for j in range(k + 2, m + 1):
            dag.add_edge(name(k, j), name(k + 1, j))

    assert dag.number_of_nodes() == (m * m + m - 2) // 2, "task count formula broken"
    assert nx.is_directed_acyclic_graph(dag), "generated graph has a cycle"
    add_random_weights(dag, weight_distribution)
    return TaskGraph.from_nx(dag)


def fft_dag(
    num_points: int,
    weight_distribution: Optional[RandomVariable] = None,
) -> TaskGraph:
    """FFT task graph for N input points (N a power of two).

    Structure from the HEFT/CPoP paper (Topcuoglu et al. 2002, Fig. 10): a
    binary tree of 2N - 1 recursive-call tasks whose N leaves feed log2(N) butterfly
    stages of N tasks each, giving (2N - 1) + N*log2(N) tasks. At butterfly
    stage l, task i depends on stage l-1 tasks i and i XOR 2^(l-1). Every
    last-stage task is an exit task; because SAGA requires a single exit task
    per graph, TaskGraph.create adds a zero-cost __super_sink__.

    Args:
        num_points (int): The FFT input size N (a power of two, at least 2).
        weight_distribution (Optional[RandomVariable]): Distribution for task
            costs and dependency sizes. Defaults to the library default.

    Returns:
        TaskGraph: The FFT task graph.

    Raises:
        ValueError: If num_points is not a power of two greater than 1.
    """
    n = num_points
    if n < 2 or n & (n - 1) != 0:
        raise ValueError(f"num_points must be a power of two >= 2, got {n}")
    depth = int(math.log2(n))

    dag = nx.DiGraph()
    for d in range(depth + 1):
        for i in range(2**d):
            dag.add_node(f"R[{d},{i}]")
            if d > 0:
                dag.add_edge(f"R[{d - 1},{i // 2}]", f"R[{d},{i}]")

    prev = [f"R[{depth},{i}]" for i in range(n)]
    for level in range(1, depth + 1):
        span = 2 ** (level - 1)
        cur = [f"B[{level},{i}]" for i in range(n)]
        for i in range(n):
            dag.add_node(cur[i])
            dag.add_edge(prev[i], cur[i])
            dag.add_edge(prev[i ^ span], cur[i])
        prev = cur

    assert dag.number_of_nodes() == (2 * n - 1) + n * depth, "task count formula broken"
    assert nx.is_directed_acyclic_graph(dag), "generated graph has a cycle"
    add_random_weights(dag, weight_distribution)
    return TaskGraph.from_nx(dag)


def compare_schedulers(network: Network, task_graph: TaskGraph) -> Dict[str, Schedule]:
    """Schedule the task graph with several schedulers and return their schedules."""
    # ETF (in general) and PEFT (on highly symmetric graphs such as FFT)
    # currently break ties by Python set-iteration order. As a result, their
    # makespans can differ between runs, even on identical instances.
    schedulers = [
        HeftScheduler(),
        CpopScheduler(),
        PEFTScheduler(),
        MinMinScheduler(),
        MaxMinScheduler(),
        SufferageScheduler(),
        ETFScheduler(),
    ]
    return {
        scheduler.name: scheduler.schedule(network, task_graph)
        for scheduler in schedulers
    }


SUPER_TASKS = ("__super_source__", "__super_sink__")


def drawable_graph(task_graph: TaskGraph) -> nx.DiGraph:
    """The task graph without SAGA's zero-cost super source/sink.

    List schedulers require a single entry task and a single exit task, and
    SAGA satisfies this by scheduling dummy tasks internally. The dummy tasks
    carry zero cost and zero-size edges and never affect the schedule. The
    drawings omit them to match the figures in the cited papers.
    """
    graph = task_graph.graph.copy()
    graph.remove_nodes_from([name for name in SUPER_TASKS if name in graph])
    return graph


def drawable_mapping(schedule: Schedule) -> Dict[str, List[ScheduledTask]]:
    """The schedule mapping without SAGA's zero-duration super tasks."""
    return {
        node: [task for task in tasks if task.name not in SUPER_TASKS]
        for node, tasks in schedule.mapping.items()
    }


def task_graph_draw_params(task_graph: TaskGraph) -> Dict[str, Any]:
    """Drawing arguments for draw_task_graph, scaled to the number of tasks.

    Larger graphs receive a larger canvas with smaller nodes and fonts, which
    keeps the task boxes from overlapping.
    """
    if len(task_graph.tasks) <= 20:
        return {
            "figsize": (12, 8),
            "node_size": 1500,
            "font_size": 13,
            "weight_font_size": 9,
        }
    return {
        "figsize": (16, 10),
        "node_size": 1200,
        "font_size": 11,
        "weight_font_size": 8,
    }


def gantt_draw_params(task_graph: TaskGraph) -> Dict[str, Any]:
    """Drawing arguments for draw_gantt, scaled to the number of tasks.

    Larger schedules receive a wider canvas with smaller fonts, which keeps
    every task label inside its bar.
    """
    if len(task_graph.tasks) <= 20:
        return {"figsize": (14, 5), "font_size": 12, "tick_font_size": 14}
    return {"figsize": (20, 6), "font_size": 10, "tick_font_size": 16}


def save_fig(ax: Axes, name: str) -> None:
    """Save the figure behind `ax` to the outputs directory, then close it."""
    fig = ax.get_figure()
    if isinstance(fig, Figure):
        fig.savefig(str(savedir / f"{name}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    # The weight pools behind RandomVariable are generated with numpy.
    # Seeding numpy makes the generated instances reproducible.
    np.random.seed(0)
    weights = UniformRandomVariable(0.5, 2.0)

    network = get_network(num_nodes=4, weight_distribution=weights)
    save_fig(draw_network(network.graph, draw_colors=False, use_latex=False), "network")

    instances = {
        "gaussian_elimination_m5": gaussian_elimination_dag(5, weights),
        "gaussian_elimination_m8": gaussian_elimination_dag(8, weights),
        "fft_n4": fft_dag(4, weights),
        "fft_n8": fft_dag(8, weights),
    }

    for label, task_graph in instances.items():
        num_super = sum(
            1
            for task in task_graph.tasks
            if task.name in ("__super_source__", "__super_sink__")
        )
        suffix = f" (+ {num_super} super task)" if num_super else ""
        print(f"\n{label}: {len(task_graph.tasks) - num_super} tasks{suffix}")
        schedules = compare_schedulers(network, task_graph)
        best_makespan = min(schedule.makespan for schedule in schedules.values())
        for sched_name, schedule in sorted(
            schedules.items(), key=lambda item: item[1].makespan
        ):
            marker = (
                "  <- best" if math.isclose(schedule.makespan, best_makespan) else ""
            )
            print(f"  {sched_name:<22} makespan {schedule.makespan:8.3f}{marker}")

        save_fig(
            draw_task_graph(
                drawable_graph(task_graph),
                use_latex=False,
                **task_graph_draw_params(task_graph),
            ),
            f"{label}_task_graph",
        )
        save_fig(
            draw_gantt(
                drawable_mapping(schedules["HeftScheduler"]),
                use_latex=False,
                **gantt_draw_params(task_graph),
            ),
            f"{label}_heft_gantt",
        )

    print(f"\nFigures saved to {savedir}")


if __name__ == "__main__":
    main()
