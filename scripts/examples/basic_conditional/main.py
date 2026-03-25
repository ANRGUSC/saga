from pathlib import Path

from saga.schedulers import HeftScheduler
from saga.utils.draw import draw_gantt, draw_task_graph, draw_mutual_exclusion_graph
from saga.utils.random_graphs import get_network, get_random_conditional_branching_dag

def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    network = get_network(num_nodes=4)

    task_graph = get_random_conditional_branching_dag(
        levels=4, branching_factor=2, conditional_parent_probability=0.9
    )

    branches = task_graph.identify_branches()
    meg = task_graph.build_mutual_exclusion_graph()
    schedule = HeftScheduler().schedule(network=network, task_graph=task_graph)

    graph_fig = draw_task_graph(task_graph.graph, use_latex=False).get_figure()
    graph_fig.savefig(output_dir / "conditional_task_graph.pdf")
    gantt_fig = draw_gantt(schedule.mapping, use_latex=False).get_figure()
    gantt_fig.savefig(output_dir / "conditional_schedule.pdf")
    meg_fig = draw_mutual_exclusion_graph(meg, use_latex=False).get_figure()
    meg_fig.savefig(output_dir / "mutual_exclusion_graph.pdf")

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
    print(f"\nMakespan: {schedule.makespan:.3f}")
    print(f"Saved PDFs in: {output_dir}")


if __name__ == "__main__":
    main()
