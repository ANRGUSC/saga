from typing import List
import networkx as nx
from ..base import TaskGraphDataset

class OutTreesDataset(TaskGraphDataset):
    """A dataset of out-trees.

    An out-tree is a directed acyclic graph (DAG) where each task has
    <branching_factor> children.

    Args:
        trees: A list of in-trees.
    """
    @staticmethod
    def generate(num: int, # pylint: disable=arguments-differ
                 num_levels: int,
                 branching_factor: int) -> "OutTreesDataset":
        """Generate a dataset of in-trees.

        Args:
            num: Number of graphs to generate.
            num_levels: Number of levels in the tree.
            branching_factor: Number of parents per node.

        Returns:
            A dataset of in-trees.
        """
        assert num > 0
        assert num_levels > 0
        assert branching_factor > 0

        trees: List[nx.DiGraph] = []
        for _ in range(num):
            # Generate the tree dag
            tree = nx.DiGraph()
            tree.add_node(0)
            for level in range(1, num_levels):
                for node in range(branching_factor ** level):
                    tree.add_node(node + branching_factor ** level)
                    for parent in range(branching_factor):
                        tree.add_edge(node + branching_factor ** (level - 1),
                                      node + branching_factor ** level)

            trees.append(tree)

        return OutTreesDataset(trees)

class InTreesDataset(TaskGraphDataset):
    """A dataset of in-trees."""
    @staticmethod
    def generate(num: int, # pylint: disable=arguments-differ
                 num_levels: int,
                 branching_factor: int) -> "InTreesDataset":
        """Generate a dataset of in-trees.

        Args:
            num: Number of graphs to generate.
            num_levels: Number of levels in the tree.
            branching_factor: Number of parents per node.

        Returns:
            A dataset of in-trees.
        """
        out_trees = OutTreesDataset.generate(num, num_levels, branching_factor)
        in_trees = []
        for tree in out_trees:
            in_trees.append(tree.reverse())

        return InTreesDataset(in_trees)

class SeriesParallelDataset(TaskGraphDataset):
    """A dataset of series-parallel graphs.

    A series-parallel graph is a directed acyclic graph (DAG) where each
    task has either 0 or 2 children.

    Args:
        graphs: A list of series-parallel graphs.
    """
    @staticmethod
    def generate(num: int, # pylint: disable=arguments-differ
                 num_levels: int) -> "SeriesParallelDataset":
        """Generate a dataset of series-parallel graphs.

        Args:
            num: Number of graphs to generate.
            num_levels: Number of levels in the tree.

        Returns:
            A dataset of series-parallel graphs.
        """
        assert num > 0
        assert num_levels > 0

        graphs: List[nx.DiGraph] = []
        for _ in range(num):
            # Generate the tree dag
            graph = nx.DiGraph()
            graph.add_node(0)
            for level in range(1, num_levels):
                for node in range(2 ** level):
                    graph.add_node(node + 2 ** level)
                    graph.add_edge(node + 2 ** (level - 1),
                                   node + 2 ** level)
                    graph.add_edge(node + 2 ** (level - 1),
                                   node + 2 ** level + 1)

            graphs.append(graph)

        return SeriesParallelDataset(graphs)