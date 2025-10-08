import networkx as nx
from typing import Optional
from saga.utils.draw import draw_task_graph


def get_possible_graphs(dag: nx.DiGraph, queue: Optional[list] = None, visited: Optional[set] = None, prob: float=1) -> list[tuple[nx.DiGraph, float]]:
    if queue is None:
        queue = [node for node in dag.nodes if dag.in_degree(node) == 0]
    if visited is None:
        visited = set()

    while queue:
        current = queue.pop(0)
        visited.add(current)

        children = list(dag.successors(current))
        is_conditional = any(dag.edges[current, child].get('conditional', False) for child in children)

        if is_conditional:
            graphs = []
            for child in children:
                graphs.extend(
                    get_possible_graphs(
                        dag, [child] + queue.copy(), visited.copy(), prob * dag.edges[current, child].get('probability', 1)
                    )
                )
            return graphs
        else:
            for child in children:
                if child not in visited and child not in queue:
                    queue.append(child)

    return [(dag.subgraph(visited).copy(), prob)] # type: ignore

def add_scaled_weights(dag: nx.DiGraph):
    possible_graphs = get_possible_graphs(dag)
    counter = {node: 0.0 for node in dag.nodes}
    for graph, prob in possible_graphs:
        for node in graph.nodes:
            counter[node] += prob
    
    for node in dag.nodes:
        dag.nodes[node]['scaled_weight'] = dag.nodes[node].get('weight', 1) * counter[node]


def main():
    dag = nx.DiGraph()

    for node in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        dag.add_node(node, weight=1)

    dag.add_edge("A", "B", weight=1, probability=1/2, conditional=True)
    dag.add_edge("A", "C", weight=1, probability=1/2, conditional=True)

    dag.add_edge("B", "D", weight=1, probability=1/3, conditional=True)
    dag.add_edge("B", "E", weight=1, probability=1/3, conditional=True)
    dag.add_edge("B", "F", weight=1, probability=1/3, conditional=True)

    dag.add_edge("C", "G", weight=1, probability=1, conditional=False)
    
    dag.add_edge("D", "H", weight=1, probability=1, conditional=False)
    dag.add_edge("E", "H", weight=1, probability=1, conditional=False)
    dag.add_edge("H", "I", weight=1, probability=1, conditional=False)
    dag.add_edge("F", "I", weight=1, probability=1, conditional=False)

    dag.add_edge("G", "J", weight=1, probability=1, conditional=False)
    dag.add_edge("I", "J", weight=1, probability=1, conditional=False)

    ax = draw_task_graph(dag, use_latex=False)
    ax.get_figure().savefig(f'example_dag.png', dpi=300)

    add_scaled_weights(dag)
    for node in dag.nodes:
        print(f"Node {node}: weight={dag.nodes[node]['weight']}, scaled_weight={dag.nodes[node]['scaled_weight']}")

if __name__ == "__main__":
    main()
    
