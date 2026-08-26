"""Print the component and combined scores on a few hand-built task graphs.

A quick sanity check that each heuristic responds to the structure it targets
(joins, branching, communication, downstream impact).
"""

from heuristics import (
    branching_score,
    communication_score,
    impact_score,
    is_super_node,
    join_val_score,
    task_score,
)

from saga import TaskGraph


def print_scores(name: str, task_graph: TaskGraph) -> None:
    print("\n" + "=" * 50)
    print(name)
    for task in task_graph.tasks:
        task_name = task.name
        if is_super_node(task_name):
            continue
        real_children = [
            edge
            for edge in task_graph.out_edges(task_name)
            if not is_super_node(edge.target)
        ]
        if not real_children:
            continue
        print(
            f"{task_name}: "
            f"comm={communication_score(task_name, task_graph):.2f}, "
            f"impact={impact_score(task_name, task_graph):.2f}, "
            f"branch={branching_score(task_name, task_graph):.2f}, "
            f"join={join_val_score(task_name, task_graph):.2f}, "
            f"final={task_score(task_name, task_graph):.2f}"
        )


def build_join_graph() -> TaskGraph:
    tasks = [
        ("A", 2.0),
        ("B", 2.0),
        ("C", 2.0),
        ("G", 2.0),
        ("H", 2.0),
        ("I", 2.0),
        ("X", 2.0),
        ("Y", 2.0),
        ("Z", 2.0),
        ("W", 2.0),
    ]
    dependencies = [
        ("A", "G", 5.0),
        ("B", "G", 10.0),
        ("C", "G", 30.0),
        ("G", "H", 20.0),
        ("G", "I", 2.0),
        ("X", "Z", 5.0),
        ("Y", "Z", 5.0),
        ("Z", "W", 5.0),
    ]
    return TaskGraph.create(tasks=tasks, dependencies=dependencies)


def build_branching_graph() -> TaskGraph:
    tasks = [
        ("A", 2.0),
        ("B", 2.0),
        ("C", 2.0),
        ("D", 2.0),
        ("E", 2.0),
        ("F", 2.0),
        ("G", 2.0),
        ("H", 2.0),
    ]
    dependencies = [
        ("A", "B", 5.0),
        ("A", "C", 5.0),
        ("A", "D", 5.0),
        ("B", "E", 5.0),
        ("C", "F", 5.0),
        ("D", "G", 5.0),
        ("F", "H", 5.0),
        ("G", "H", 5.0),
    ]
    return TaskGraph.create(tasks=tasks, dependencies=dependencies)


def build_communication_graph() -> TaskGraph:
    tasks = [
        ("A", 2.0),
        ("B", 2.0),
        ("C", 2.0),
        ("D", 2.0),
        ("E", 2.0),
        ("F", 2.0),
        ("G", 2.0),
        ("X", 2.0),
        ("Y", 2.0),
        ("Z", 2.0),
        ("W", 2.0),
        ("Q", 2.0),
    ]
    dependencies = [
        ("A", "B", 5.0),
        ("B", "C", 20.0),
        ("B", "D", 20.0),
        ("B", "E", 20.0),
        ("D", "F", 5.0),
        ("E", "G", 5.0),
        ("X", "Y", 60.0),
        ("Y", "Z", 5.0),
        ("Y", "W", 5.0),
        ("W", "Q", 5.0),
    ]
    return TaskGraph.create(tasks=tasks, dependencies=dependencies)


def build_impact_graph() -> TaskGraph:
    tasks = [
        ("A", 2.0),
        ("B", 2.0),
        ("C", 2.0),
        ("D", 2.0),
        ("E", 2.0),
        ("F", 2.0),
        ("G", 2.0),
        ("H", 2.0),
        ("I", 2.0),
        ("X", 2.0),
        ("Y", 2.0),
    ]
    dependencies = [
        ("A", "B", 5.0),
        ("B", "C", 5.0),
        ("B", "D", 5.0),
        ("C", "E", 5.0),
        ("C", "F", 5.0),
        ("D", "G", 5.0),
        ("E", "H", 5.0),
        ("G", "I", 5.0),
        ("X", "Y", 5.0),
    ]
    return TaskGraph.create(tasks=tasks, dependencies=dependencies)


def main() -> None:
    print_scores("JOIN GRAPH", build_join_graph())
    print_scores("BRANCHING GRAPH", build_branching_graph())
    print_scores("COMMUNICATION GRAPH", build_communication_graph())
    print_scores("IMPACT GRAPH", build_impact_graph())


if __name__ == "__main__":
    main()
