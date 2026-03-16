from pathlib import Path

from saga.schedulers import HeftScheduler
from saga.utils.draw import draw_gantt, draw_task_graph
from saga.utils.random_graphs import get_network, get_random_conditional_branching_dag

def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    network = get_network(num_nodes=4)
    task_graph = get_random_conditional_branching_dag(
        levels=3, branching_factor=2, conditional_parent_probability=0.4
    )
    task_to_group = task_graph.identify_conditional_groups()
    schedule = HeftScheduler().schedule(network=network, task_graph=task_graph)
    graph_fig = draw_task_graph(task_graph.graph, use_latex=False).get_figure()
    if graph_fig is not None:
        graph_fig.savefig(output_dir / "conditional_task_graph.pdf")
    gantt_fig = draw_gantt(schedule.mapping, use_latex=False).get_figure()
    if gantt_fig is not None:
        gantt_fig.savefig(output_dir / "conditional_schedule.pdf")
    print(f"Tasks: {len(task_graph.tasks)}, Edges: {len(task_graph.dependencies)}")
    print("Task groups (-1 means non-conditional):")
    for task_name in sorted(task_to_group):
        print(f"  {task_name}: {task_to_group[task_name]}")
    print("Edge probabilities:")
    for edge in sorted(task_graph.dependencies):
        print(f"  {edge.source} -> {edge.target}: p={edge.probability:.2f}")
    print(f"Makespan: {schedule.makespan:.3f}")
    print(f"Saved PDFs in: {output_dir}")


if __name__ == "__main__":
    main()
