"""Basic conditional task graph (CTG) scheduling example.

A CTG has *conditional* edges: at a fork, only one successor branch runs at
runtime. Tasks on different branches are therefore *mutually exclusive* and may
share a processor/time slot. This example builds a small CTG, shows its traces
and mutual-exclusion structure, schedules it with the conditional HEFT/CPoP
schedulers (which overlap mutually exclusive tasks), and splits the overlapping
schedule back into one Gantt chart per trace.
"""

import logging
import pathlib

from saga import Network, Schedule
from saga.conditional import (
    ConditionalTaskGraph,
    ConditionalTaskGraphEdge,
    extract_traces_with_recalculation,
)
from saga.schedulers.cheft import CheftScheduler
from saga.schedulers.ccpop import CCpopScheduler
from saga.utils.draw import (
    draw_gantt,
    draw_mutual_exclusion_graph,
    draw_network,
    draw_task_graph,
)

logging.basicConfig(level=logging.INFO)

thisdir = pathlib.Path(__file__).parent.absolute()
savedir = thisdir / "outputs"
savedir.mkdir(exist_ok=True)


def get_instance() -> tuple[Network, ConditionalTaskGraph]:
    """A two-node network and a CTG that conditionally takes one of two branches.

    ``A`` forks conditionally to ``B`` (p=0.6) or ``C`` (p=0.4); the branches run
    ``B -> D`` or ``C -> E`` and rejoin at ``F``. The two traces are
    ``A, B, D, F`` and ``A, C, E, F``, so ``{B, D}`` and ``{C, E}`` are mutually
    exclusive.
    """
    network = Network.create(
        nodes=[("v1", 1.0), ("v2", 1.0)],
        edges=[("v1", "v2", 1.0)],
    )
    task_graph = ConditionalTaskGraph.create(
        tasks=[
            ("A", 1.0), ("B", 1.0), ("C", 1.0),
            ("D", 1.0), ("E", 1.0), ("F", 1.0),
        ],
        dependencies=[
            ConditionalTaskGraphEdge(source="A", target="B", size=1.0, probability=0.6),
            ConditionalTaskGraphEdge(source="A", target="C", size=1.0, probability=0.4),
            ConditionalTaskGraphEdge(source="B", target="D", size=1.0),
            ConditionalTaskGraphEdge(source="C", target="E", size=1.0),
            ConditionalTaskGraphEdge(source="D", target="F", size=1.0),
            ConditionalTaskGraphEdge(source="E", target="F", size=1.0),
        ],
    )
    return network, task_graph


def _save(ax, name: str) -> None:
    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(str(savedir / f"{name}.png"))  # type: ignore


def draw_instance(network: Network, task_graph: ConditionalTaskGraph) -> None:
    _save(draw_task_graph(task_graph.graph, use_latex=False), "task_graph")
    _save(draw_network(network.graph, draw_colors=False, use_latex=False), "network")
    _save(
        draw_mutual_exclusion_graph(task_graph.build_mutual_exclusion_graph()),
        "mutual_exclusion_graph",
    )


def report_traces(task_graph: ConditionalTaskGraph) -> None:
    print("Traces:")
    for trace in task_graph.identify_traces_detailed():
        tasks = ", ".join(trace["tasks"])
        print(f"  {trace['name']}: [{tasks}]  (p={trace['probability']})")

    meg_edges = sorted(tuple(sorted(e)) for e in task_graph.build_mutual_exclusion_graph().edges)
    print(f"Mutually exclusive (may overlap) pairs: {meg_edges}")


def run_scheduler(scheduler, network: Network, task_graph: ConditionalTaskGraph, label: str) -> Schedule:
    schedule = scheduler.schedule(network, task_graph)
    _save(draw_gantt(schedule.mapping), f"{label}_overlapping_gantt")

    # Expected makespan over the trace distribution: split the overlapping
    # schedule per trace, then weight each trace's makespan by its probability.
    prob_by_name = {t["name"]: t["probability"] for t in task_graph.identify_traces_detailed()}
    per_trace = extract_traces_with_recalculation(schedule)
    expected_makespan = 0.0
    print(f"{label}: overlapping makespan = {schedule.makespan:.3f}")
    for name, mapping in per_trace.items():
        trace_makespan = max(
            (t.end for tasks in mapping.values() for t in tasks), default=0.0
        )
        expected_makespan += prob_by_name.get(name, 0.0) * trace_makespan
        safe_name = name.replace(":", "").replace(" ", "_")
        _save(draw_gantt(mapping), f"{label}_trace_{safe_name}_gantt")
        print(f"    {name}: makespan = {trace_makespan:.3f}")
    print(f"{label}: expected makespan over traces = {expected_makespan:.3f}")
    return schedule


def main() -> None:
    network, task_graph = get_instance()
    draw_instance(network, task_graph)
    report_traces(task_graph)
    run_scheduler(CheftScheduler(), network, task_graph, "cheft")
    run_scheduler(CCpopScheduler(), network, task_graph, "ccpop")


if __name__ == "__main__":
    main()
