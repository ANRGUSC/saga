import json
from saga.data import serialize_graph
from saga.experiment.benchmarking.parametric.components import UpwardRanking, GreedyInsert, ParametricScheduler
from saga.utils.random_graphs import add_random_weights, get_branching_dag, get_network
import networkx as nx
import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
from dgl.data import DGLDataset
import pandas as pd

from saga.experiment import datadir, resultsdir, outputdir

def prepare_dataset():
    LEVELS = 3
    BRANCHING_FACTOR = 2
    NUM_NODES = 10
    NUM_PROBLEMS = 100

    dataset = []
    for i in range(NUM_PROBLEMS):
        print(f"Progress: {i+1/NUM_PROBLEMS*100:.2f}%" + " " * 10, end="\r")
        task_graph = add_random_weights(get_branching_dag(levels=LEVELS, branching_factor=BRANCHING_FACTOR))
        network = add_random_weights(get_network(num_nodes=NUM_NODES))

        topological_sorts = list(nx.all_topological_sorts(task_graph))
        best_makespan = float("inf")
        best_topological_sort = None
        for topological_sort in topological_sorts:
            scheduler = ParametricScheduler(
                initial_priority=lambda *_: topological_sort,
                insert_task=GreedyInsert(
                    append_only=False,
                    compare="EFT",
                    critical_path=False
                )
            )

            schedule = scheduler.schedule(network.copy(), task_graph.copy())
            makespan = max(task.end for tasks in schedule.values() for task in tasks)

            if makespan < best_makespan:
                best_makespan = makespan
                best_topological_sort = topological_sort

        dataset.append(
            {
                "task_graph": serialize_graph(task_graph),
                "network": serialize_graph(network),
                "topological_sort": best_topological_sort
            }
        )

    print("Progress: 100.00%")

    savedir = datadir / "ml"
    savedir.mkdir(exist_ok=True, parents=True)
    dataset_path = savedir / "data.json"
    dataset_path.write_text(json.dumps(dataset, indent=4))

class MLSchedulingDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="karate_club")

    def process(self):
        dataset = json.loads((datadir / "ml" / "data.json").read_text())

        nodes_data = pd.read_csv("./members.csv")
        edges_data = pd.read_csv("./interactions.csv")
        node_features = torch.from_numpy(nodes_data["Age"].to_numpy())
        node_labels = torch.from_numpy(
            nodes_data["Club"].astype("category").cat.codes.to_numpy()
        )
        edge_features = torch.from_numpy(edges_data["Weight"].to_numpy())
        edges_src = torch.from_numpy(edges_data["Src"].to_numpy())
        edges_dst = torch.from_numpy(edges_data["Dst"].to_numpy())

        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes=nodes_data.shape[0]
        )
        self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels
        self.graph.edata["weight"] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True
        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["test_mask"] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


