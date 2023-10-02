from saga.schedulers.data.wfcommons import gen_workflows, get_wfcommon_cloud

def main():
    network = get_wfcommon_cloud("chameleon")
    task_graph = gen_workflows(num=1, recipe_name="cycles")[0]

    min_node_weight = min(network.nodes[node]["weight"] for node in network.nodes)
    avg_node_weight = sum(network.nodes[node]["weight"] for node in network.nodes) / len(network.nodes)
    max_node_weight = max(network.nodes[node]["weight"] for node in network.nodes)

    min_link_weight = min(network.edges[edge]["weight"] for edge in network.edges)
    avg_link_weight = sum(network.edges[edge]["weight"] for edge in network.edges) / len(network.edges)
    max_link_weight = max(network.edges[edge]["weight"] for edge in network.edges)

    min_task_weight = min(task_graph.nodes[node]["weight"] for node in task_graph.nodes)
    avg_task_weight = sum(task_graph.nodes[node]["weight"] for node in task_graph.nodes) / len(task_graph.nodes)
    max_task_weight = max(task_graph.nodes[node]["weight"] for node in task_graph.nodes)

    min_edge_weight = min(task_graph.edges[edge]["weight"] for edge in task_graph.edges)
    avg_edge_weight = sum(task_graph.edges[edge]["weight"] for edge in task_graph.edges) / len(task_graph.edges)
    max_edge_weight = max(task_graph.edges[edge]["weight"] for edge in task_graph.edges)

    max_comp_time = max_task_weight / min_node_weight
    min_comp_time = min_task_weight / max_node_weight
    avg_comp_time = avg_task_weight / avg_node_weight

    max_comm_time = max_edge_weight / min_link_weight
    min_comm_time = min_edge_weight / max_link_weight
    avg_comm_time = avg_edge_weight / avg_link_weight

    print("max_comp_time", max_comp_time)
    print("min_comp_time", min_comp_time)
    print("avg_comp_time", avg_comp_time)

    print("max_comm_time", max_comm_time)
    print("min_comm_time", min_comm_time)
    print("avg_comm_time", avg_comm_time)

    


if __name__ == "__main__":
    main()