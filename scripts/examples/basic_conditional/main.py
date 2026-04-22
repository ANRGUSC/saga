from pathlib import Path

import matplotlib.pyplot as plt

from saga.schedulers import HeftScheduler
from saga.utils.draw import draw_gantt, draw_task_graph
from saga.utils.random_graphs import get_network, get_random_conditional_branching_dag
from saga.conditional import extract_traces_with_recalculation


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    network = get_network(num_nodes=4)

    task_graph = get_random_conditional_branching_dag(
        levels=3, branching_factor=2, conditional_parent_probability=0.5
    )

    traces = task_graph.identify_traces()
    meg = task_graph.build_mutual_exclusion_graph()
    schedule = HeftScheduler().schedule(network=network, task_graph=task_graph)
    trace_schedules = extract_traces_with_recalculation(schedule)

    # --- Save full-graph visualisations ---
    draw_task_graph(task_graph.graph, use_latex=False).get_figure().savefig(
        output_dir / "conditional_task_graph.pdf"
    )
    draw_gantt(schedule.mapping, use_latex=False).get_figure().savefig(
        output_dir / "conditional_schedule.pdf"
    )

    # --- Save per-trace visualisations ---
    for trace_name, trace_mapping in trace_schedules.items():
        safe_name = trace_name.replace(" ", "_").replace(":", "")
        trace_tasks = {t.name for tasks in trace_mapping.values() for t in tasks}
        trace_subgraph = task_graph.graph.subgraph(trace_tasks).copy()

        draw_task_graph(trace_subgraph, use_latex=False).get_figure().savefig(
            output_dir / f"trace_taskgraph_{safe_name}.pdf"
        )
        draw_gantt(trace_mapping, use_latex=False).get_figure().savefig(
            output_dir / f"trace_schedule_{safe_name}.pdf"
        )
        plt.close("all")

    # --- Console summary ---
    print(f"Tasks: {len(task_graph.tasks)}, Edges: {len(task_graph.dependencies)}")
    print("\nEdge probabilities:")
    for edge in sorted(task_graph.dependencies):
        prob = edge.probability if hasattr(edge, "probability") else 1.0
        print(f"  {edge.source} -> {edge.target}: p={prob:.2f}")
    print(f"\nTraces ({len(traces)}):")
    for name, tasks in traces:
        print(f"  {name}: {tasks}")
    print(f"\nMutual exclusion graph — nodes: {sorted(meg.nodes)}")
    print(f"Mutually exclusive pairs ({meg.number_of_edges()}):")
    for a, b in sorted(meg.edges):
        print(f"  {a} <-> {b}")
    print(f"\nOverlapping schedule makespan: {schedule.makespan:.3f}")
    for trace_name, trace_mapping in trace_schedules.items():
        trace_makespan = max(
            (t.end for tasks in trace_mapping.values() for t in tasks), default=0.0
        )
        print(f"  {trace_name} makespan (recalculated): {trace_makespan:.3f}")
    print(f"\nSaved PDFs in: {output_dir}")


if __name__ == "__main__":
    main()
