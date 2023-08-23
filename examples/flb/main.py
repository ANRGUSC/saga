from saga.schedulers.flb import FLBScheduler
import networkx as nx

tg_dict = {'edges': {('A', 'B'): 0.7586716402378093,
           ('B', '__dst__'): 1e-09,
           ('C', '__dst__'): 1e-09,
           ('D', '__dst__'): 1e-09,
           ('__src__', 'A'): 1e-09,
           ('__src__', 'C'): 1e-09,
           ('__src__', 'D'): 1e-09},
 'nodes': {'A': 1e-09,
           'B': 0.7392714364829512,
           'C': 0.8616683782519285,
           'D': 0.058020598190548583,
           '__dst__': 1e-09,
           '__src__': 1e-09}}
nw_dict = {'edges': {(0, 0): 1000000000.0,
           (0, 1): 1,
           (0, 2): 1,
           (0, 3): 1,
           (1, 1): 1000000000.0,
           (1, 2): 1,
           (1, 3): 1,
           (2, 2): 1000000000.0,
           (2, 3): 1,
           (3, 3): 1000000000.0},
 'nodes': {0: 1, 1: 1, 2: 1, 3: 1}}

def main():
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from(tg_dict['nodes'].keys())
    task_graph.add_edges_from(tg_dict['edges'].keys())
    for node, weight in tg_dict['nodes'].items():
        task_graph.nodes[node]['weight'] = weight
    for edge, weight in tg_dict['edges'].items():
        task_graph.edges[edge]['weight'] = weight

    network = nx.Graph()
    network.add_nodes_from(nw_dict['nodes'].keys())
    network.add_edges_from(nw_dict['edges'].keys())
    for node, weight in nw_dict['nodes'].items():
        network.nodes[node]['weight'] = weight
    for edge, weight in nw_dict['edges'].items():
        network.edges[edge]['weight'] = weight


    scheduler = FLBScheduler()
    schedule = scheduler.schedule(network, task_graph)
    print(schedule)


if __name__ == "__main__":
    main()