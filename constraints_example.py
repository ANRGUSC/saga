"""
Demonstration of the SAGA Constraints API.

Shows how to use AllowedNodes and ForbiddenNodes constraints with both the
ParametricScheduler and concrete schedulers (HEFT, MCT, MinMin, MaxMin).

Usage:
    python constraints_example.py
    python constraints_example.py --gantt
"""

import argparse
import pathlib
import sys

sys.path.insert(0, "src")

import networkx as nx

from saga import Network, TaskGraph
from saga.constraints import AllowedNodes, ForbiddenNodes, Constraints
from saga.schedulers.heft import HeftScheduler
from saga.schedulers.mct import MCTScheduler
from saga.schedulers.minmin import MinMinScheduler
from saga.schedulers.maxmin import MaxMinScheduler
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    GreedyInsert,
    GreedyInsertCompareFuncs,
    UpwardRanking,
)

thisdir = pathlib.Path(__file__).parent


# ---------------------------------------------------------------------------
# Problem instance
# ---------------------------------------------------------------------------

def get_instance():
    """Three-node network, four-task diamond DAG."""
    network = nx.Graph()
    network.add_node("v_1", weight=1)
    network.add_node("v_2", weight=2)
    network.add_node("v_3", weight=4)
    network.add_edge("v_1", "v_2", weight=2)
    network.add_edge("v_1", "v_3", weight=1)
    network.add_edge("v_2", "v_3", weight=4)

    task_graph = nx.DiGraph()
    task_graph.add_node("t_1", weight=4)
    task_graph.add_node("t_2", weight=3)
    task_graph.add_node("t_3", weight=2)
    task_graph.add_node("t_4", weight=1)
    task_graph.add_edge("t_1", "t_2", weight=2)
    task_graph.add_edge("t_1", "t_3", weight=1)
    task_graph.add_edge("t_2", "t_4", weight=3)
    task_graph.add_edge("t_3", "t_4", weight=3)

    return Network.from_nx(network), TaskGraph.from_nx(task_graph)


def get_pinned_instance():
    """Same instance with t_1 pinned to v_1 via the legacy pinned_to field."""
    network = nx.Graph()
    network.add_node("v_1", weight=1)
    network.add_node("v_2", weight=2)
    network.add_node("v_3", weight=4)
    network.add_edge("v_1", "v_2", weight=2)
    network.add_edge("v_1", "v_3", weight=1)
    network.add_edge("v_2", "v_3", weight=4)

    task_graph = nx.DiGraph()
    task_graph.add_node("t_1", weight=4, pinned_to="v_1")  # legacy pin
    task_graph.add_node("t_2", weight=3)
    task_graph.add_node("t_3", weight=2)
    task_graph.add_node("t_4", weight=1)
    task_graph.add_edge("t_1", "t_2", weight=2)
    task_graph.add_edge("t_1", "t_3", weight=1)
    task_graph.add_edge("t_2", "t_4", weight=3)
    task_graph.add_edge("t_3", "t_4", weight=3)

    return Network.from_nx(network), TaskGraph.from_nx(task_graph)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_schedule(schedule, label: str):
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Makespan : {schedule.makespan:.4f}")
    for node_name, tasks in sorted(schedule.mapping.items()):
        if tasks:
            task_strs = ", ".join(
                f"{t.name}[{t.start:.2f}–{t.end:.2f}]" for t in tasks
            )
            print(f"  {node_name:6s} : {task_strs}")


def save_gantt(schedule, label: str, savedir: pathlib.Path, xmax: float | None = None):
    try:
        from saga.utils.draw import draw_gantt
        import matplotlib.pyplot as plt

        ax = draw_gantt(schedule.mapping, use_latex=False, xmax=xmax)
        ax.set_title(label)
        fname = label.lower().replace(" ", "_").replace("/", "_") + ".png"
        plt.tight_layout()
        plt.savefig(savedir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {savedir / fname}")
    except Exception as e:
        print(f"  Could not save Gantt chart: {e}")


# ---------------------------------------------------------------------------
# Scenario 1 — no constraints (baseline)
# ---------------------------------------------------------------------------

def run_unconstrained(gantt: bool, savedir: pathlib.Path):
    print("\n" + "="*55)
    print("SCENARIO 1 — No Constraints (baseline)")
    print("="*55)
    network, task_graph = get_instance()

    heft = HeftScheduler().schedule(network, task_graph)
    print_schedule(heft, "HEFT (unconstrained)")

    parametric = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(append_only=False, compare=GreedyInsertCompareFuncs.EFT),
    ).schedule(network, task_graph)
    print_schedule(parametric, "Parametric/EFT/UpwardRanking (unconstrained)")

    if gantt:
        xmax = max(heft.makespan, parametric.makespan)
        save_gantt(heft, "HEFT unconstrained", savedir, xmax)
        save_gantt(parametric, "Parametric unconstrained", savedir, xmax)


# ---------------------------------------------------------------------------
# Scenario 2 — AllowedNodes via explicit Constraints object
# ---------------------------------------------------------------------------

def run_allowed_nodes(gantt: bool, savedir: pathlib.Path):
    print("\n" + "="*55)
    print("SCENARIO 2 — AllowedNodes Constraints")
    print("  t_1 pinned to v_1 | t_4 restricted to {v_2, v_3}")
    print("="*55)
    network, task_graph = get_instance()

    constraints = Constraints(constraints=[
        AllowedNodes(task="t_1", nodes=["v_1"]),           # exact pin
        AllowedNodes(task="t_4", nodes=["v_2", "v_3"]),    # restrict to subset
    ])

    for SchCls, label in [
        (HeftScheduler,    "HEFT"),
        (MCTScheduler,     "MCT"),
        (MinMinScheduler,  "MinMin"),
        (MaxMinScheduler,  "MaxMin"),
    ]:
        # Concrete schedulers currently derive constraints from pinned_to.
        # Pass a task_graph that carries the same restrictions via pinned_to
        # OR construct the scheduler with an overridden constraints object.
        # Here we demonstrate the ParametricScheduler path (explicit constraints)
        # and the legacy pinned_to path for the concrete schedulers.
        pass

    # Parametric scheduler: constraints passed explicitly
    parametric = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(append_only=False, compare=GreedyInsertCompareFuncs.EFT),
    ).schedule(network, task_graph, constraints=constraints)
    print_schedule(parametric, "Parametric/EFT/UpwardRanking (AllowedNodes)")

    # HEFT: constraints are automatically read from pinned_to — demonstrate
    # that the legacy path still works identically for single-node pins.
    network2, task_graph2 = get_pinned_instance()
    heft_pinned = HeftScheduler().schedule(network2, task_graph2)
    print_schedule(heft_pinned, "HEFT (legacy pinned_to on t_1 → v_1)")

    if gantt:
        save_gantt(parametric, "Parametric AllowedNodes", savedir)
        save_gantt(heft_pinned, "HEFT legacy pin", savedir)


# ---------------------------------------------------------------------------
# Scenario 3 — ForbiddenNodes constraints
# ---------------------------------------------------------------------------

def run_forbidden_nodes(gantt: bool, savedir: pathlib.Path):
    print("\n" + "="*55)
    print("SCENARIO 3 — ForbiddenNodes Constraints")
    print("  t_2 forbidden from v_1 | t_3 forbidden from v_3")
    print("="*55)
    network, task_graph = get_instance()

    constraints = Constraints(constraints=[
        ForbiddenNodes(task="t_2", nodes=["v_1"]),
        ForbiddenNodes(task="t_3", nodes=["v_3"]),
    ])

    parametric = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(append_only=False, compare=GreedyInsertCompareFuncs.EFT),
    ).schedule(network, task_graph, constraints=constraints)
    print_schedule(parametric, "Parametric/EFT/UpwardRanking (ForbiddenNodes)")

    if gantt:
        save_gantt(parametric, "Parametric ForbiddenNodes", savedir)


# ---------------------------------------------------------------------------
# Scenario 4 — Mixed AllowedNodes + ForbiddenNodes, all schedulers
# ---------------------------------------------------------------------------

def run_mixed_all_schedulers(gantt: bool, savedir: pathlib.Path):
    print("\n" + "="*55)
    print("SCENARIO 4 — Mixed Constraints, all schedulers")
    print("  t_1 pinned to v_1 | t_2 forbidden from v_1")
    print("="*55)
    network, task_graph = get_instance()

    constraints = Constraints(constraints=[
        AllowedNodes(task="t_1", nodes=["v_1"]),
        ForbiddenNodes(task="t_2", nodes=["v_1"]),
    ])

    schedulers = {
        "HEFT":    HeftScheduler(),
        "MCT":     MCTScheduler(),
        "MinMin":  MinMinScheduler(),
        "MaxMin":  MaxMinScheduler(),
        "Parametric/EFT": ParametricScheduler(
            initial_priority=UpwardRanking(),
            insert_task=GreedyInsert(
                append_only=False, compare=GreedyInsertCompareFuncs.EFT
            ),
        ),
    }

    # Concrete schedulers (HEFT/MCT/MinMin/MaxMin) consume constraints via
    # Constraints.from_task_graph(). To pass explicit constraints to them,
    # we build a task_graph that encodes AllowedNodes as pinned_to, and pass
    # a modified task_graph.  For ForbiddenNodes on concrete schedulers,
    # use ParametricScheduler which accepts constraints directly.
    #
    # For this demonstration, we use ParametricScheduler variants for full
    # constraint support, and HEFT/MCT via the pinned_to migration path.

    results = {}
    for name, sched in schedulers.items():
        if hasattr(sched, "schedule"):
            try:
                if isinstance(sched, ParametricScheduler):
                    result = sched.schedule(network, task_graph, constraints=constraints)
                else:
                    # Concrete schedulers: build a task_graph with pinned_to
                    # for the AllowedNodes (single-pin) constraint only.
                    pinned_nx = nx.DiGraph()
                    for task in task_graph.tasks:
                        attrs = {"weight": task.cost}
                        # Find if this task has an AllowedNodes single-pin
                        for c in constraints.constraints:
                            if c.task == task.name and isinstance(c, AllowedNodes) and len(c.nodes) == 1:
                                attrs["pinned_to"] = c.nodes[0]
                        pinned_nx.add_node(task.name, **attrs)
                    for dep in task_graph.dependencies:
                        pinned_nx.add_edge(dep.source, dep.target, weight=dep.size)
                    from saga import TaskGraph as TG
                    pinned_tg = TG.from_nx(pinned_nx)
                    result = sched.schedule(network, pinned_tg)
                results[name] = result
                print_schedule(result, f"{name} (mixed constraints)")
            except Exception as e:
                print(f"  {name}: ERROR — {e}")

    if gantt and results:
        xmax = max(s.makespan for s in results.values())
        for name, sched in results.items():
            save_gantt(sched, f"Mixed_{name}", savedir, xmax)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAGA Constraints API demo")
    parser.add_argument("--gantt", action="store_true", help="Save Gantt chart PNGs")
    args = parser.parse_args()

    savedir = thisdir / "constraint_outputs"
    if args.gantt:
        savedir.mkdir(exist_ok=True)
        print(f"Gantt charts will be saved to: {savedir}")

    run_unconstrained(args.gantt, savedir)
    run_allowed_nodes(args.gantt, savedir)
    run_forbidden_nodes(args.gantt, savedir)
    run_mixed_all_schedulers(args.gantt, savedir)

    print("\nDone.")


if __name__ == "__main__":
    main()
