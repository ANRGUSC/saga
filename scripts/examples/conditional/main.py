"""
Main experiment runner for conditional workflow testing.
"""
import pathlib
import networkx as nx
from generate_dags import create_test_conditional_dags, save_dag_as_pdf
from test_conditional_ranking import compare_ranking_approaches, get_all_execution_paths

def main():
    """Run conditional workflow experiments
        Test weight scaling for consistency
    """

    
    # Create a simple network
    network = nx.Graph()
    network.add_nodes_from([0, 1, 2])
    for node in network.nodes:
        network.nodes[node]["weight"] = 1.0
    
    # Add edges with communication speeds
    for u, v in [(0, 1), (0, 2), (1, 2)]:
        network.add_edge(u, v, weight=1.0)
    
    # Test on various DAGs
    test_cases = create_test_conditional_dags()
    dag2 = test_cases[1][1]
    save_dag_as_pdf(dag2, "dag2")
    # run simple test
    static_schedules = get_all_execution_paths(dag2)
    # Loop through all items and save each one
    for i, (path_name, dag) in enumerate(static_schedules.items()):
        # Save each DAG with a unique name
        save_dag_as_pdf(dag, f"static_schedule_{i}_{path_name}")

if __name__ == "__main__":
    main()