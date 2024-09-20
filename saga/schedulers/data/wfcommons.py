import importlib
import json
import pathlib
import pickle
import random
import tempfile
from functools import lru_cache
from itertools import product
from typing import Dict, Iterable, List, Optional, Set, Tuple, Type, Union, Callable

import git
import networkx as nx
from scipy import stats
from wfcommons.common.workflow import Workflow, Task
from wfcommons import wfchef
from wfcommons.wfchef.recipes import (BlastRecipe, BwaRecipe, CyclesRecipe,
                                      EpigenomicsRecipe, GenomeRecipe,
                                      MontageRecipe, SeismologyRecipe,
                                      SoykbRecipe, SrasearchRecipe)
from wfcommons.wfgen.abstract_recipe import WorkflowRecipe
from wfcommons.wfgen.generator import WorkflowGenerator
from wfcommons.wfchef.wfchef_abstract_recipe import WfChefWorkflowRecipe

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

clouds = {
    "chameleon": {
        "repo": "https://github.com/wfcommons/pegasus-instances.git",
        "glob": "*/chameleon-cloud/*.json"
    }
}
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
    if cloud_name not in clouds:
        raise ValueError(f"Cloud {cloud_name} not found. Available clouds: {clouds.keys()}")

    repo = clouds[cloud_name]["repo"]
    glob = clouds[cloud_name]["glob"]

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

pegasus_workflows = {
    wfname: {
        "repo": "https://github.com/wfcommons/pegasus-instances.git",
        "glob": f"{wfname}/chameleon-cloud/*.json"
    } for wfname in ["1000genome", "cycles", "epigenomics", "montage", "seismology", "soykb", "srasearch"]
}
makeflow_workflows = {
    wfname: {
        "repo": "https://github.com/wfcommons/makeflow-instances.git",
        "glob": f"{wfname}/chameleon-cloud/*.json"
    } for wfname in ["blast", "bwa"]
}
workflow_instances = {**pegasus_workflows, **makeflow_workflows}
workflow_instances["genome"] = workflow_instances["1000genome"]
def get_real_workflows(wfname: str) -> List[Workflow]:
    """Get workflows representing the specified wfname.

    Args:
        wfname (str): The name of the workflow.

    Returns:
        List[Workflow]: The workflows representing the workflow.

    Raises:
        ValueError: If the wfname is not found.
    """
    if wfname not in workflow_instances:
        raise ValueError(f"Workflow {wfname} not found. Available workflows: {workflow_instances.keys()}")

    repo = workflow_instances[wfname]["repo"]
    glob = workflow_instances[wfname]["glob"]

    path_repo = download_repo(repo)
    workflows: List[Workflow] = []
    for path in pathlib.Path(path_repo).glob(glob):
        workflows.append(trace_to_digraph(path))
    return workflows
    


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


def get_networks(num: int,
                 cloud_name: str,
                 network_speed: float = 100) -> List[nx.Graph]:
    """Generate a random networkx graph.

    Since Chameleon cloud uses a shared file-system for communication and network speeds are very hight
    (10-100 Gbps), the communication bottleneck is the SSD speed. Default network speed is based on 
    specification of SSD in Chameleon cloud at time of writing (11/19/2023).
    Source: https://www.disctech.com/Seagate-ST2000NX0273-2000GB-SAS-Hard-Drive

    Args:
        num (int): The number of networks to generate.
        cloud_name (str): The name of the cloud.
        network_speed (float, optional): The speed of the network in MegaBytes per second. Defaults to 136.

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
            network.add_edge(src, dst, weight=network_speed if src != dst else 1e9)

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

    children: Dict[str, Set[str]] = {}
    for task in trace["workflow"]["tasks"]:
        for parent in task.get("parents", []):
            children.setdefault(parent, set()).add(task["name"])

    task_graph = nx.DiGraph()
    outputs: Dict[str, Dict[str, float]] = {}
    input_files: Dict[str, Set[str]] = {}
    for task in trace["workflow"]["tasks"]:
        if "children" not in task:
            task["children"] = children.get(task["name"], set())

        runtime = task.get('runtime', task.get('runtimeInSeconds'))
        if runtime is None:
            raise ValueError(f"Task {task['name']} has no 'runtime' or 'runtimeInSeconds' attribute. {list(task.keys())}. {path}")
        task_graph.add_node(task["name"], weight=max(1e-9, runtime))

        for io_file in task["files"]:
            if io_file["link"] == "output":
                size_in_bytes = io_file.get("sizeInBytes", io_file.get("size"))
                if size_in_bytes is None:
                    raise ValueError(f"File {io_file['name']} has no 'sizeInBytes' or 'size' attribute. {path}")
                outputs.setdefault(task["name"], {})[io_file["name"]] = size_in_bytes / 1e6 # convert to MB
            elif io_file["link"] == "input":
                input_files.setdefault(task["name"], set()).add(io_file["name"])

    for task in trace["workflow"]["tasks"]:
        for child in task.get("children", []):
            weight = 0
            for input_file in input_files.get(child, []):
                weight += outputs[task["name"]].get(input_file, 0)
            task_graph.add_edge(task["name"], child, weight=max(1e-9, weight))

    return task_graph

def get_workflow_task_info(recipe_name: str) -> Dict:
    """Get the task type statistics for the given recipe.
    
    Args:
        recipe_name (str): The name of the recipe.

    Returns:
        Dict: The task type statistics.
    """
    path = pathlib.Path(wfchef.__file__).parent.joinpath('recipes', recipe_name, 'task_type_stats.json')
    if not path.exists():
        raise ValueError(f"Recipe {recipe_name} not found. Available recipes: {recipes.keys()}")
    return json.loads(path.read_text(encoding="utf-8"))

def build_workflow(recipe: WfChefWorkflowRecipe,
                   workflow_name: Optional[str] = None,
                   graph: Optional[nx.DiGraph] = None) -> Workflow:
    """Generate a synthetic workflow instance.

    NOTE: This function is a copy of the build_workflow method in wfcommons.wfchef.wfchef_abstract_recipe 
    modified to allow providing a graph so that only the weights need to be generated.

    Args:
        recipe (WorkflowRecipe): The recipe to generate the workflow from.
        workflow_name (Optional[str], optional): The workflow name. Defaults to None.
        graph (Optional[nx.DiGraph], optional): The graph to generate the workflow from. Defaults to None.

    Returns:
        Workflow: A synthetic workflow instance object.
    """
    workflow = Workflow(name=recipe.name + "-synthetic-instance" if not workflow_name else workflow_name,
                        makespan=0)
    if graph is None:
        graph = recipe.generate_nx_graph()
    else:
        graph = graph.copy()

    task_names = {}
    for node in graph.nodes:
        if node in ["SRC", "DST"]:
            continue
        node_type = graph.nodes[node]["type"]
        task_name = recipe._generate_task_name(node_type)
        task = recipe._generate_task(node_type, task_name)
        workflow.add_node(task_name, task=task)

        task_names[node] = task_name

    # tasks dependencies
    for (src, dst) in graph.edges:
        if src in ["SRC", "DST"] or dst in ["SRC", "DST"]:
            continue
        workflow.add_edge(task_names[src], task_names[dst])

        if task_names[src] not in recipe.tasks_children:
            recipe.tasks_children[task_names[src]] = []
        if task_names[dst] not in recipe.tasks_parents:
            recipe.tasks_parents[task_names[dst]] = []

        recipe.tasks_children[task_names[src]].append(task_names[dst])
        recipe.tasks_parents[task_names[dst]].append(task_names[src])

    # find leaf tasks
    leaf_tasks = []
    for node_name in workflow.nodes:
        task: Task = workflow.nodes[node_name]['task']
        if task.name not in recipe.tasks_children:
            leaf_tasks.append(task)

    for task in leaf_tasks:
        recipe._generate_task_files(task)

    workflow.nxgraph = graph
    recipe.workflows.append(workflow)
    return workflow

def get_workflows(num: int,
                  recipe_name: str,
                  vary_weights_only: bool = False) -> List[nx.DiGraph]:
    """Generate a list of network, task graph pairs for the given recipe.

    Args:
        num (int): The number of task graphs to generate.
        recipe_name (str): The name of the recipe.
        vary_weights_only (bool, optional): Whether to vary only the weights of the tasks. Defaults to False.

    Returns:
        List[nx.DiGraph]: The list of task graphs.
    """
    if recipe_name not in recipes:
        raise ValueError(f"Recipe {recipe_name} not found. Available recipes: {recipes.keys()}")

    task_graphs: List[nx.DiGraph] = []
    graph = None
    for _ in range(num):
        num_tasks = random.randint(*get_num_task_range(recipe_name))
        recipe = recipes[recipe_name](num_tasks=num_tasks)
        workflow = build_workflow(recipe, graph=graph)
        if vary_weights_only:
            graph = workflow.nxgraph.copy()
        with tempfile.NamedTemporaryFile() as tmp:
            workflow.write_json(tmp.name)
            task_graphs.append(trace_to_digraph(tmp.name))

    return task_graphs

def test():
    """Test the module."""
    task_graph = get_workflows(1, "1000genome")[0]
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
