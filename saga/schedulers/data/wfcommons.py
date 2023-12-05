import importlib
import json
import pathlib
import pickle
import random
import tempfile
from functools import lru_cache
from itertools import product
from typing import Dict, Iterable, List, Tuple, Type, Union, Callable

import git
import networkx as nx
from scipy import stats
from wfcommons.common.workflow import Workflow
from wfcommons import wfchef
from wfcommons.wfchef.recipes import (BlastRecipe, BwaRecipe, CyclesRecipe,
                                      EpigenomicsRecipe, GenomeRecipe,
                                      MontageRecipe, SeismologyRecipe,
                                      SoykbRecipe, SrasearchRecipe)
from wfcommons.wfgen.abstract_recipe import WorkflowRecipe
from wfcommons.wfgen.generator import WorkflowGenerator

recipes: Dict[str, Type[WorkflowRecipe]] = {
    'epigenomics': EpigenomicsRecipe,
    'montage': MontageRecipe,
    'cycles': CyclesRecipe,
    'seismology': SeismologyRecipe,
    'soykb': SoykbRecipe,
    'srasearch': SrasearchRecipe,
    'genome': GenomeRecipe,
    'blast': BlastRecipe,
    'bwa': BwaRecipe
}

@lru_cache(maxsize=None)
def get_num_task_range(recipe_name: str) -> Tuple[int, int]:
    """Get the minimum and maximum number of tasks for the given recipe.

    Args:
        recipe_name (str): The name of the recipe.

    Returns:
        Tuple[int, int]: The minimum and maximum number of tasks.
    """
    # get file_location of recipe
    recipe_module = importlib.import_module(f"wfcommons.wfchef.recipes.{recipe_name}")
    path = pathlib.Path(recipe_module.__file__).parent

    min_tasks = float("inf")
    max_tasks = 0
    for path in path.glob("microstructures/*/base_graph.pickle"):
        graph = pickle.loads(path.read_bytes())
        min_tasks = min(min_tasks, graph.number_of_nodes())
        max_tasks = max(max_tasks, graph.number_of_nodes())
    return min_tasks, max_tasks


instances = {
    "chameleon": {
        "repo": "https://github.com/wfcommons/pegasus-instances.git",
        "glob": "*/chameleon-cloud/*.json"
    }
}
def download_repo(repo: str) -> pathlib.Path:
    """Download the given to ~/.saga/data/<repo_name> if it does not exist.

    Args:
        repo (str): The git repository to download.

    Returns:
        pathlib.Path: The path to the downloaded repository.

    Raises:
        GitCommandError: If the git command fails.
    """
    # remove .git if at end of repo
    repo_name = pathlib.Path(repo).stem
    path = pathlib.Path.home() / ".saga" / "data" / repo_name
    if path.exists():
        return path

    git.Repo.clone_from(repo, path, progress=git.remote.RemoteProgress())
    return path


def get_real_networks(cloud_name: str) -> List[nx.Graph]:
    """Get graphs representing the specified cloud_name.

    Args:
        cloud_name (str): The name of the cloud.

    Returns:
        List[nx.Graph]: The networkx graphs representing the cloud.

    Raises:
        ValueError: If the cloud_name is not found.
    """
    # git clone - print progress
    if cloud_name not in instances:
        raise ValueError(f"Cloud {cloud_name} not found. Available clouds: {instances.keys()}")

    repo = instances[cloud_name]["repo"]
    glob = instances[cloud_name]["glob"]

    path_repo = download_repo(repo)
    networks: Dict[str, nx.Graph] = {} # hash -> graph
    for path in pathlib.Path(path_repo).glob(glob):
        workflow = json.loads(path.read_text(encoding="utf-8"))
        network = nx.Graph()
        for machine in workflow["workflow"]["machines"]:
            network.add_node(machine["nodeName"], weight=machine["cpu"]["speed"])

        network_hash = hash(frozenset({(node, network.nodes[node]["weight"]) for node in network.nodes}))

        # normalize machine speeds to average speed
        avg_speed = sum(network.nodes[node]["weight"] for node in network.nodes) / len(network.nodes)
        for node in network.nodes:
            network.nodes[node]["weight"] /= avg_speed

        for (src, dst) in product(network.nodes, network.nodes):
            # unlimited bandwidth since i/o is integrated into task weights
            network.add_edge(src, dst, weight=1e9)

        networks[network_hash] = network

    return list(networks.values())

def get_best_fit(data: Iterable) -> Callable[[int], List[float]]:
    """Get a function that generates random numbers from the best fit distribution.

    Args:
        data (Iterable): The data to fit.

    Returns:
        Callable[[int], List[float]]: A function that generates random numbers from the best fit distribution.
            Values are clipped to the min and max of the data.

    Raises:
        ValueError: If the data is empty.
    """
    distribution_types = ['norm', 'expon', 'lognorm', 'gamma', 'beta', 'uniform']
    fit_results = {}
    for dist_type in distribution_types:
        params = getattr(stats, dist_type).fit(data)
        D, p_value = stats.kstest(data, dist_type, args=params)
        fit_results[dist_type] = {
            "params": params,
            "D": D,
            "p_value": p_value
        }

    best_dist = min(fit_results, key=lambda k: fit_results[k]["D"])
    best_fit_params = fit_results[best_dist]["params"]

    min_data = min(data)
    max_data = max(data)

    return lambda num: [
        max(min_data, min(max_data, float(value)))
        for value in getattr(stats, best_dist).rvs(size=num, *best_fit_params)
    ]


def get_networks(num: int, cloud_name: str) -> List[nx.Graph]:
    """Generate a random networkx graph.

    Args:
        num (int): The number of networks to generate.
        cloud_name (str): The name of the cloud.


    Returns:
        List[nx.Graph]: The list of networkx graphs.
    """
    real_networks = get_real_networks(cloud_name)
    get_num_nodes = get_best_fit([network.number_of_nodes() for network in real_networks])
    get_node_speed = get_best_fit([network.nodes[node]["weight"] for network in real_networks
                                   for node in network.nodes])

    all_num_nodes = list(map(int, get_num_nodes(num)))
    networks: List[nx.Graph] = []
    for num_nodes in all_num_nodes:
        network = nx.Graph()
        node_speeds = get_node_speed(num_nodes)
        for i in range(num_nodes):
            network.add_node(i, weight=max(1e-9, node_speeds[i]))

        for (src, dst) in product(network.nodes, network.nodes):
            # unlimited bandwidth since i/o is integrated into task weights
            network.add_edge(src, dst, weight=1e9)

        networks.append(network)

    return networks


def trace_to_digraph(path: Union[str, pathlib.Path]) -> nx.DiGraph:
    """Convert a trace to a networkx DiGraph.

    Args:
        path (Union[str, pathlib.Path]): The path to the trace.

    Returns:
        nx.DiGraph: The networkx DiGraph.
    """
    trace = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

    task_graph = nx.DiGraph()
    for task in trace["workflow"]["tasks"]:
        runtime = task.get('runtime', task.get('runtimeInSeconds'))
        if runtime is None:
            raise ValueError(f"Task {task['name']} has no 'runtime' or 'runtimeInSeconds' attribute.")
        task_graph.add_node(task["name"], weight=max(1e-9, runtime))

    for task in trace["workflow"]["tasks"]:
        for child in task.get("children", []):
            task_graph.add_edge(task["name"], child, weight=1e-9)

    return task_graph

def get_workflow_task_info(recipe_name: str) -> Dict:
    path = pathlib.Path(wfchef.__file__).parent.joinpath('recipes', recipe_name, 'task_type_stats.json')
    if not path.exists():
        raise ValueError(f"Recipe {recipe_name} not found. Available recipes: {recipes.keys()}")
    return json.loads(path.read_text(encoding="utf-8"))

def get_workflows(num: int, recipe_name: str) -> List[nx.DiGraph]:
    """Generate a list of network, task graph pairs for the given recipe.

    Args:
        num (int): The number of task graphs to generate.
        recipe_name (str): The name of the recipe.

    Returns:
        List[nx.DiGraph]: The list of task graphs.
    """
    if recipe_name not in recipes:
        raise ValueError(f"Recipe {recipe_name} not found. Available recipes: {recipes.keys()}")

    task_graphs: List[nx.DiGraph] = []
    for _ in range(num):
        num_tasks = random.randint(*get_num_task_range(recipe_name))
        recipe = recipes[recipe_name](num_tasks=num_tasks)
        generator = WorkflowGenerator(recipe)
        workflow = generator.build_workflow()
        with tempfile.NamedTemporaryFile() as tmp:
            workflow.write_json(tmp.name)
            task_graphs.append(trace_to_digraph(tmp.name))

    return task_graphs

def test():
    """Test the module."""
    task_graph = get_workflows(1, "blast")[0]
    # print node and edge attributes
    for node in task_graph.nodes:
        print(node, task_graph.nodes[node])
    for edge in task_graph.edges:
        print(edge, task_graph.edges[edge])

    network = get_networks(1, "chameleon")[0]
    # print node and edge attributes
    for node in network.nodes:
        print(node, network.nodes[node])
    for edge in network.edges:
        print(edge, network.edges[edge])

if __name__ == "__main__":
    test()
