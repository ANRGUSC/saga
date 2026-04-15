from pathlib import Path

import matplotlib.pyplot as plt

from saga.schedulers import HeftScheduler
from saga.utils.draw import draw_gantt, draw_task_graph
from saga.utils.random_graphs import get_network, get_random_conditional_branching_dag
from saga.conditional import extract_branches_with_recalculation


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    network = get_network(num_nodes=4)

    task_graph = get_random_conditional_branching_dag(
        levels=3, branching_factor=2, conditional_parent_probability=0.5
    )

    branches = task_graph.identify_branches()
    meg = task_graph.build_mutual_exclusion_graph()
    schedule = HeftScheduler().schedule(network=network, task_graph=task_graph)
    branch_schedules = extract_branches_with_recalculation(schedule)

    # --- Save full-graph visualisations ---
    draw_task_graph(task_graph.graph, use_latex=False).get_figure().savefig(
        output_dir / "conditional_task_graph.pdf"
    )
    draw_gantt(schedule.mapping, use_latex=False).get_figure().savefig(
        output_dir / "conditional_schedule.pdf"
    )

    # --- Save per-branch visualisations ---
    for branch_name, branch_mapping in branch_schedules.items():
        safe_name = branch_name.replace(" ", "_").replace(":", "")
        branch_tasks = {t.name for tasks in branch_mapping.values() for t in tasks}
        branch_subgraph = task_graph.graph.subgraph(branch_tasks).copy()

        draw_task_graph(branch_subgraph, use_latex=False).get_figure().savefig(
            output_dir / f"branch_taskgraph_{safe_name}.pdf"
        )
        draw_gantt(branch_mapping, use_latex=False).get_figure().savefig(
            output_dir / f"branch_schedule_{safe_name}.pdf"
        )
        plt.close("all")

    # --- Console summary ---
    print(f"Tasks: {len(task_graph.tasks)}, Edges: {len(task_graph.dependencies)}")
    print("\nEdge probabilities:")
    for edge in sorted(task_graph.dependencies):
        prob = edge.probability if hasattr(edge, "probability") else 1.0
        print(f"  {edge.source} -> {edge.target}: p={prob:.2f}")
    print(f"\nBranches ({len(branches)}):")
    for name, tasks in branches:
        print(f"  {name}: {tasks}")
    print(f"\nMutual exclusion graph — nodes: {sorted(meg.nodes)}")
    print(f"Mutually exclusive pairs ({meg.number_of_edges()}):")
    for a, b in sorted(meg.edges):
        print(f"  {a} <-> {b}")
    print(f"\nOverlapping schedule makespan: {schedule.makespan:.3f}")
    for branch_name, branch_mapping in branch_schedules.items():
        branch_makespan = max(
            (t.end for tasks in branch_mapping.values() for t in tasks), default=0.0
        )
        print(f"  {branch_name} makespan (recalculated): {branch_makespan:.3f}")
    print(f"\nSaved PDFs in: {output_dir}")


if __name__ == "__main__":
    main()
