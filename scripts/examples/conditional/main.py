"""Test ConditionalHeftScheduler with static DAGs from CTG."""
import sys
import pathlib
import networkx as nx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.parent / "src"))

from bfs import get_possible_graphs, add_scaled_weights
from saga.schedulers.conditional_heft import ConditionalHeftScheduler
from saga.utils.draw import draw_gantt, draw_task_graph
from generate_dags import save_dag_as_pdf

def main():
    # Create CTG copied from bfs.py
    ctg = nx.DiGraph()
    for node in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        ctg.add_node(node, weight=10)
    ctg.add_edge("A", "B", weight=5, probability=0.5, conditional=True)
    ctg.add_edge("A", "C", weight=5, probability=0.5, conditional=True)
    ctg.add_edge("B", "D", weight=3, probability=1/3, conditional=True)
    ctg.add_edge("B", "E", weight=3, probability=1/3, conditional=True)
    ctg.add_edge("B", "F", weight=3, probability=1/3, conditional=True)
    ctg.add_edge("C", "G", weight=4, probability=1, conditional=False)
    ctg.add_edge("D", "H", weight=2, probability=1, conditional=False)
    ctg.add_edge("E", "H", weight=2, probability=1, conditional=False)
    ctg.add_edge("H", "I", weight=3, probability=1, conditional=False)
    ctg.add_edge("F", "I", weight=3, probability=1, conditional=False)
    ctg.add_edge("G", "J", weight=5, probability=1, conditional=False)
    ctg.add_edge("I", "J", weight=5, probability=1, conditional=False)
    

    add_scaled_weights(ctg)
    static_dags = get_possible_graphs(ctg)
    
    # Create network 
    network = nx.Graph()
    network.add_nodes_from([1, 2, 3], weight=1)
    network.add_edges_from([(1, 2), (1, 3), (2, 3)], weight=1)
    network.add_edges_from([(1, 1), (2, 2), (3, 3)], weight=1e9)  # self-loops 
    
    # schedule static dags
    scheduler = ConditionalHeftScheduler()
    save_dag_as_pdf(ctg, "CTG")
    for i, (dag, prob) in enumerate(static_dags):
        # copy weights from CTG to static dags
        for node in dag.nodes:
            dag.nodes[node]['scaled_weight'] = ctg.nodes[node]['scaled_weight']
        
        # schedule
        schedule = scheduler.schedule(network, dag)
        makespan = max(task.end for tasks in schedule.values() for task in tasks)
        
        # Debug
        print(f"Path {i+1}: prob={prob}, makespan={makespan}, nodes={list(dag.nodes)}")
        
        # save
        save_dag_as_pdf(dag, f"static dag {i+1}, prob = {prob:.3f}")
        import matplotlib.pyplot as plt
        ax = draw_gantt(schedule, dag)
        ax.set_title(f"Path {i+1}: prob={prob:.3f}, makespan={makespan:.2f}")
        ax.get_figure().savefig(f"schedule_{i+1}.pdf", bbox_inches='tight')
        plt.close(ax.get_figure())

if __name__ == "__main__":
    main()
