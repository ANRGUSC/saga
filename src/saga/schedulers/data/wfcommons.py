import importlib
import json
import logging
import pathlib
import pickle
import pprint
import random
import shutil
import tempfile
from functools import lru_cache
from itertools import product
import traceback
from typing import Dict, Iterable, List, Optional, Set, Tuple, Type, Union, Callable

import git
import networkx as nx
from scipy import stats
import numpy as np
import scipy
from saga.utils.random_variable import RandomVariable

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
    'cycles': CyclesRecipe,
    'epigenomics': EpigenomicsRecipe,
    'montage': MontageRecipe,
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

def generate_rvs(distribution: Dict[str, Union[str, List[float]]],
                 min_value: float,
                 max_value: float,
                 num: int) -> List[float]:
    if not distribution or distribution == "None":
        return [min_value] * num

    params = distribution['params']
    kwargs = params[:-2]
    rvs = np.clip(getattr(scipy.stats, distribution['name']).rvs(*kwargs, loc=params[-2], scale=params[-1], size=num), 0.1, np.inf)
    rvs *= max_value
    return rvs

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
                 network_speed: float = 100,
                 rv_weights: bool = False) -> List[nx.Graph]:
    """Generate a random networkx graph.

    Since Chameleon cloud uses a shared file-system for communication and network speeds are very hight
    (10-100 Gbps), the communication bottleneck is the SSD speed. Default network speed is based on 
    specification of SSD in Chameleon cloud at time of writing (11/19/2023).
    Source: https://www.disctech.com/Seagate-ST2000NX0273-2000GB-SAS-Hard-Drive

    Args:
        num (int): The number of networks to generate.
        cloud_name (str): The name of the cloud.
        network_speed (float, optional): The speed of the network in MegaBytes per second. Defaults to 136.
        rv_weights (bool, optional): Whether to use random variables for the weights. Defaults to False.

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
            if rv_weights:
                network.add_node(i, weight=RandomVariable(node_speeds))
            else:
                network.add_node(i, weight=max(1e-9, node_speeds[i]))

        for (src, dst) in product(network.nodes, network.nodes):
            network.add_edge(src, dst, weight=network_speed if src != dst else 1e9)

        networks.append(network)

    return networks


def trace_to_digraph(path: Union[str, pathlib.Path],
                     recipe_name: str,
                     rv_weights: bool = False) -> nx.DiGraph:
    trace = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    task_stats = get_workflow_task_info(recipe_name)

    workflow = nx.DiGraph()

    parents: Dict[str, Set[str]] = {}
    file_producers: Dict[str, str] = {}
    task_names = {}
    for task in trace["workflow"]["specification"]["tasks"]:
        task_names[task["id"]] = task["name"]
        for child in task["children"]:
            parents.setdefault(child, set()).add(task["id"])
        for file_id in task["outputFiles"]:
            file_producers[file_id] = task["id"]

    for task in trace["workflow"]["execution"]["tasks"]:
        task_name = task_names[task["id"]]
        if rv_weights:
            # name = task_stats[task_name]["runtime"]["distribution"]["name"]
            # params = task_stats[task_name]["runtime"]["distribution"]["params"]
            # rvs: float = getattr(stats, name).rvs(*params, size=100)
            rvs = generate_rvs(
                distribution=task_stats[task_name]["runtime"]["distribution"],
                min_value=task_stats[task_name]["runtime"]["min"],
                max_value=task_stats[task_name]["runtime"]["max"],
                num=100
            )
            rvs = np.clip(rvs, 1e-9, None)
            runtime = RandomVariable(rvs)
        else:
            runtime = task.get('runtime', task.get('runtimeInSeconds'))

        workflow.add_node(task["id"], weight=max(1e-9, runtime), color=task_name)
    
    file_sizes = {}
    for file in trace["workflow"]["specification"]["files"]:
        if file["id"] not in file_producers:
            continue
        producer_id = file_producers[file["id"]]
        task_name = task_names[producer_id]
        input_type = file["id"][36:]
        min_weight = task_stats[task_name]["output"][input_type]["min"]
        max_weight = task_stats[task_name]["output"][input_type]["max"]
        if rv_weights:
            if task_stats[task_name]["output"][input_type]["distribution"]:
                # name = task_stats[task_name]["output"][input_type]["distribution"]["name"]
                # params = task_stats[task_name]["output"][input_type]["distribution"]["params"]
                # rvs: float = getattr(stats, name).rvs(*params, size=100)
                rvs = generate_rvs(
                    distribution=task_stats[task_name]["output"][input_type]["distribution"],
                    min_value=min_weight,
                    max_value=max_weight,
                    num=100
                )
                rvs = np.clip(rvs, min_weight, max_weight)
                dep_rv = RandomVariable(rvs)
                file_sizes[file["id"]] = dep_rv
            else:
                file_sizes[file["id"]] = RandomVariable([min_weight] * 100)

        else:
            file_sizes[file["id"]] = file["sizeInBytes"]

    for task in trace["workflow"]["specification"]["tasks"]:
        for child in task["children"]:
            inputs_from_task = [
                file_id for file_id in task["inputFiles"]
                if file_producers.get(file_id) == task["id"]
            ]
            weight = RandomVariable([1e-9]*100) if rv_weights else 0.0
            for input_file in inputs_from_task:
                if input_file in file_sizes:
                    weight += file_sizes[input_file]

            workflow.add_edge(task["id"], child, weight=max(1e-9, weight))
    return workflow

# def trace_to_digraph(path: Union[str, pathlib.Path],
#                      recipe_name: str,
#                      rv_weights: bool = False) -> nx.DiGraph:
#     """Convert a trace to a networkx DiGraph.

#     Args:
#         path (Union[str, pathlib.Path]): The path to the trace.

#     Returns:
#         nx.DiGraph: The networkx DiGraph.
#     """
#     trace = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

#     if float(trace["schemaVersion"]) >= 1.5:
#         return trace_to_digraph_15(path, recipe_name, rv_weights=rv_weights)

#     task_stats = get_workflow_task_info(recipe_name)

#     children: Dict[str, Set[str]] = {}
#     for task in trace["workflow"]["tasks"]:
#         for parent in task.get("parents", []):
#             children.setdefault(parent, set()).add(task["name"])

#     task_graph = nx.DiGraph()
#     outputs: Dict[str, Dict[str, float]] = {}
#     input_files: Dict[str, Set[str]] = {}
#     for task in trace["workflow"]["tasks"]:
#         if "children" not in task:
#             task["children"] = children.get(task["name"], set())

#         if rv_weights:
#             # name = task_stats[task["name"]]["runtime"]["distribution"]["name"]
#             # params = task_stats[task["name"]]["runtime"]["distribution"]["params"]
#             # rvs: float = getattr(stats, name).rvs(*params, size=100)
#             rvs = generate_rvs(
#                 distribution=task_stats[task["name"]]["runtime"]["distribution"],
#                 min_value=task_stats[task["name"]]["runtime"]["min"],
#                 max_value=task_stats[task["name"]]["runtime"]["max"],
#                 num=100
#             )
#             rvs = np.clip(rvs, 1e-9, None)
#             runtime = RandomVariable(rvs)
#         else:
#             runtime = task.get('runtime', task.get('runtimeInSeconds'))
#         if runtime is None:
#             raise ValueError(f"Task {task['name']} has no 'runtime' or 'runtimeInSeconds' attribute. {list(task.keys())}. {path}")
#         task_graph.add_node(task["name"], weight=max(1e-9, runtime))

#         for io_file in task["files"]:
#             if io_file["link"] == "output":
#                 if rv_weights:
#                     # name = task_stats[task["name"]]["output"]["distribution"]["name"]
#                     # params = task_stats[task["name"]]["output"]["distribution"]["params"]
#                     # rvs: float = getattr(stats, name).rvs(loc=params[-2], scale=params[-1], size=100)
#                     rvs = generate_rvs(
#                         distribution=task_stats[task["name"]]["output"][io_file["name"]]["distribution"],
#                         min_value=task_stats[task["name"]]["output"][io_file["name"]]["min"],
#                         max_value=task_stats[task["name"]]["output"][io_file["name"]]["max"],
#                         num=100
#                     )
#                     rvs = np.clip(rvs, 1e-9, None)
#                     size_in_bytes = RandomVariable(rvs)
#                 else:
#                     size_in_bytes = io_file.get("sizeInBytes", io_file.get("size"))
#                 if size_in_bytes is None:
#                     raise ValueError(f"File {io_file['name']} has no 'sizeInBytes' or 'size' attribute. {path}")
#                 outputs.setdefault(task["name"], {})[io_file["name"]] = size_in_bytes / 1e6 # convert to MB
#             elif io_file["link"] == "input":
#                 input_files.setdefault(task["name"], set()).add(io_file["name"])

#     for task in trace["workflow"]["tasks"]:
#         for child in task.get("children", []):
#             weight = 0
#             for input_file in input_files.get(child, []):
#                 weight += outputs[task["name"]].get(input_file, 0)
#             task_graph.add_edge(task["name"], child, weight=max(1e-9, weight))

#     return task_graph

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


def get_workflows(num: int,
                  recipe_name: str,
                  rv_weights: bool = False,
                  max_size_multiplier: int = None) -> List[nx.DiGraph]:
    """Generate a list of network, task graph pairs for the given recipe.

    Args:
        num (int): The number of task graphs to generate.
        recipe_name (str): The name of the recipe.
        rv_weights (bool, optional): Whether to use random variables for the weights. Defaults to False.

    Returns:
        List[nx.DiGraph]: The list of task graphs.
    """
    if recipe_name not in recipes:
        raise ValueError(f"Recipe {recipe_name} not found. Available recipes: {recipes.keys()}")

    task_graphs: List[nx.DiGraph] = []
    graph = None
    for _ in range(num):
        min_tasks, max_tasks = get_num_task_range(recipe_name)
        if max_size_multiplier is not None:
            max_tasks = max_size_multiplier * min_tasks
        num_tasks = random.randint(min_tasks, max_tasks)
        recipe = recipes[recipe_name](num_tasks=num_tasks)
        generator = WorkflowGenerator(recipe)
        workflow = generator.build_workflow()
        with tempfile.NamedTemporaryFile() as tmp:
            workflow.write_json(tmp.name)
            # shutil.copy(tmp.name, "trace.json")
            task_graphs.append(trace_to_digraph(tmp.name, recipe_name, rv_weights=rv_weights))

    return task_graphs


def get_wfcommons_instance(recipe_name: str,
                           ccr: float,
                           estimate_method: Callable[[RandomVariable, bool], float] = lambda x, is_speed: x.mean(),
                           max_size_multiplier: int = 2) -> Tuple[nx.Graph, nx.DiGraph]:
    workflow = get_workflows(num=1, recipe_name=recipe_name, rv_weights=True, max_size_multiplier=max_size_multiplier)[0]

    # rename weight attribute to weight_rv
    for node in workflow.nodes:
        weight_rv: RandomVariable = workflow.nodes[node]["weight"]
        workflow.nodes[node]["weight_rv"] = weight_rv
        workflow.nodes[node]["weight_estimate"] = max(1e-9, estimate_method(weight_rv, is_speed=True))
        workflow.nodes[node]["weight_actual"] = max(1e-9, weight_rv.sample())
        workflow.nodes[node]["weight"] = workflow.nodes[node]["weight_estimate"]

    for (u, v) in workflow.edges:
        weight_rv: RandomVariable = workflow.edges[u, v]["weight"]
        workflow.edges[u, v]["weight_rv"] = weight_rv
        workflow.edges[u, v]["weight_estimate"] = max(1e-9, estimate_method(weight_rv, is_speed=True))
        workflow.edges[u, v]["weight_actual"] = max(1e-9, weight_rv.sample())
        workflow.edges[u, v]["weight"] = workflow.edges[u, v]["weight_estimate"]

    # add src and dst task
    workflow.add_node("SRC", weight=1e-9, weight_estimate=1e-9, weight_actual=1e-9, weight_rv=RandomVariable([1e-9]))
    workflow.add_node("DST", weight=1e-9, weight_estimate=1e-9, weight_actual=1e-9, weight_rv=RandomVariable([1e-9]))
    for node in workflow.nodes:
        if node not in ["SRC", "DST"] and not workflow.in_degree(node):
            workflow.add_edge("SRC", node, weight=1e9, weight_estimate=1e9, weight_actual=1e9, weight_rv=RandomVariable([1e9]))
    for node in workflow.nodes:
        if node not in ["SRC", "DST"] and not workflow.out_degree(node):
            workflow.add_edge(node, "DST", weight=1e9, weight_estimate=1e9, weight_actual=1e9, weight_rv=RandomVariable([1e9]))

    # network_speed = RandomVariable(samples=np.clip(np.random.normal(1, 0.3, 100), 1e-9, np.inf))
    # ccr = (avg_task_cost / avg_node_speed) / (avg_dep_cost / avg_comm_speed)

    network: nx.Graph = get_networks(
        num=1,
        cloud_name="chameleon",
        network_speed=RandomVariable([1]*100), # dummy value, will be adjusted later for CCR
        rv_weights=True
    )[0]

    for node in network.nodes:
        weight_rv: RandomVariable = network.nodes[node]["weight"]
        network.nodes[node]["weight_rv"] = weight_rv
        network.nodes[node]["weight_estimate"] = max(1e-9, estimate_method(weight_rv, is_speed=True) if isinstance(weight_rv, RandomVariable) else weight_rv)
        network.nodes[node]["weight_actual"] = max(1e-9, weight_rv.sample() if isinstance(weight_rv, RandomVariable) else weight_rv)
        network.nodes[node]["weight"] = network.nodes[node]["weight_estimate"]

    # adjust network edges to match CCR
    avg_task_cost = np.mean([workflow.nodes[node]["weight_actual"] for node in workflow.nodes])
    avg_dep_cost = np.mean([workflow.edges[u, v]["weight_actual"] for u, v in workflow.edges])
    avg_node_speed = np.mean([network.nodes[node]["weight_actual"] for node in network.nodes])
    avg_comm_speed = ccr * avg_dep_cost / (avg_task_cost / avg_node_speed)
    for (u, v) in network.edges:
        weight_rv: RandomVariable = RandomVariable([avg_comm_speed] * 100)
        network.edges[u, v]["weight"] = weight_rv
        network.edges[u, v]["weight_rv"] = weight_rv
        network.edges[u, v]["weight_estimate"] = max(1e-9, estimate_method(weight_rv, is_speed=True) if isinstance(weight_rv, RandomVariable) else weight_rv)
        network.edges[u, v]["weight_actual"] = max(1e-9, weight_rv.sample() if isinstance(weight_rv, RandomVariable) else weight_rv)
        network.edges[u, v]["weight"] = network.edges[u, v]["weight_estimate"]

    # ccr = (avg_task_cost / avg_node_speed) / (avg_dep_cost / avg_comm_speed)
    # avg_node_speed_new = (avg_task_cost / ccr) / (avg_dep_cost / avg_comm_speed)
    # avg_node_speed_multiplier = avg_node_speed_new / avg_comm_speed
    # for node in network.nodes:
    #     network.nodes[node]["weight_rv"] *= avg_node_speed_multiplier
    #     network.nodes[node]["weight_estimate"] *= avg_node_speed_multiplier
    #     network.nodes[node]["weight_actual"] *= avg_node_speed_multiplier
    #     network.nodes[node]["weight"] = network.nodes[node]["weight_estimate"]

    return network, workflow

