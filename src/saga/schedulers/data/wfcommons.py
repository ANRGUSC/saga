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
from git.remote import RemoteProgress
import networkx as nx
from scipy import stats
import numpy as np
import scipy

from saga import Network, TaskGraph
from saga.utils.random_variable import RandomVariable

# wfcommons is an optional dependency (does not install on Windows)
try:
    from wfcommons import wfchef
    from wfcommons.wfchef.recipes import (
        BlastRecipe,
        BwaRecipe,
        CyclesRecipe,
        EpigenomicsRecipe,
        GenomeRecipe,
        MontageRecipe,
        SeismologyRecipe,
        SoykbRecipe,
        SrasearchRecipe,
    )
    from wfcommons.wfgen.abstract_recipe import WorkflowRecipe
    from wfcommons.wfgen.generator import WorkflowGenerator

    WFCOMMONS_AVAILABLE = True
except ImportError:
    WFCOMMONS_AVAILABLE = False
    wfchef = None  # type: ignore
    WorkflowRecipe = None  # type: ignore
    WorkflowGenerator = None  # type: ignore
    BlastRecipe = None  # type: ignore
    BwaRecipe = None  # type: ignore
    CyclesRecipe = None  # type: ignore
    EpigenomicsRecipe = None  # type: ignore
    GenomeRecipe = None  # type: ignore
    MontageRecipe = None  # type: ignore
    SeismologyRecipe = None  # type: ignore
    SoykbRecipe = None  # type: ignore
    SrasearchRecipe = None  # type: ignore


def _check_wfcommons_available() -> None:
    """Check if wfcommons is available and raise an error with installation instructions if not."""
    if not WFCOMMONS_AVAILABLE:
        raise ImportError(
            "wfcommons is required for this functionality but is not installed. "
            "wfcommons is not supported on Windows and is automatically installed on other platforms. "
            "If you are on Linux/macOS, try reinstalling: pip install --force-reinstall anrg-saga"
        )


def _get_recipes() -> Dict[str, Type]:
    """Get the available wfcommons recipes. Returns empty dict if wfcommons is not installed."""
    if not WFCOMMONS_AVAILABLE:
        return {}
    return {
        "cycles": CyclesRecipe,
        "epigenomics": EpigenomicsRecipe,
        "montage": MontageRecipe,
        "seismology": SeismologyRecipe,
        "soykb": SoykbRecipe,
        "srasearch": SrasearchRecipe,
        "genome": GenomeRecipe,
        "blast": BlastRecipe,
        "bwa": BwaRecipe,
    }


recipes: Dict[str, Type] = _get_recipes()


@lru_cache(maxsize=None)
def get_num_task_range(recipe_name: str) -> Tuple[int, int]:
    """Get the minimum and maximum number of tasks for the given recipe.

    Args:
        recipe_name (str): The name of the recipe.

    Returns:
        Tuple[int, int]: The minimum and maximum number of tasks.
    """
    _check_wfcommons_available()
    recipe_module = importlib.import_module(f"wfcommons.wfchef.recipes.{recipe_name}")
    if recipe_module.__file__ is None:
        raise ValueError(
            f"Recipe {recipe_name} not found. Available recipes: {recipes.keys()}"
        )
    path = pathlib.Path(recipe_module.__file__).parent

    min_tasks = float("inf")
    max_tasks = 0
    for path in path.glob("microstructures/*/base_graph.pickle"):
        graph: nx.Graph = pickle.loads(path.read_bytes())
        min_tasks = min(min_tasks, graph.number_of_nodes())
        max_tasks = max(max_tasks, graph.number_of_nodes())
    return int(min_tasks), max_tasks


def generate_rvs(
    distribution: Dict[str, str | List[float]],
    min_value: float,
    max_value: float,
    num: int,
) -> List[float]:
    if not distribution or distribution == "None":
        return [min_value] * num

    params = distribution["params"]
    kwargs = params[:-2]
    dist_name: str = str(distribution["name"])
    rvs = np.clip(
        getattr(scipy.stats, dist_name).rvs(
            *kwargs, loc=params[-2], scale=params[-1], size=num
        ),
        a_min=0.1,
        a_max=np.inf,
    )
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

    git.Repo.clone_from(repo, path, progress=RemoteProgress())  # type: ignore
    return path


clouds = {
    "chameleon": {
        "repo": "https://github.com/wfcommons/pegasus-instances.git",
        "glob": "*/chameleon-cloud/*.json",
    }
}


@lru_cache(maxsize=None)
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
        raise ValueError(
            f"Cloud {cloud_name} not found. Available clouds: {clouds.keys()}"
        )

    repo = clouds[cloud_name]["repo"]
    glob = clouds[cloud_name]["glob"]

    path_repo = download_repo(repo)
    networks: Dict[str, nx.Graph] = {}  # hash -> graph
    for path in pathlib.Path(path_repo).glob(glob):
        workflow = json.loads(path.read_text(encoding="utf-8"))
        network = nx.Graph()
        for machine in workflow["workflow"]["machines"]:
            network.add_node(machine["nodeName"], weight=machine["cpu"]["speed"])

        network_hash = str(
            hash(
                frozenset(
                    {(node, network.nodes[node]["weight"]) for node in network.nodes}
                )
            )
        )

        # normalize machine speeds to average speed
        avg_speed = sum(network.nodes[node]["weight"] for node in network.nodes) / len(
            network.nodes
        )
        for node in network.nodes:
            network.nodes[node]["weight"] /= avg_speed

        for src, dst in product(network.nodes, network.nodes):
            # unlimited bandwidth since i/o is integrated into task weights
            network.add_edge(src, dst, weight=1e9)

        networks[network_hash] = network

    return list(networks.values())


pegasus_workflows = {
    wfname: {
        "repo": "https://github.com/wfcommons/pegasus-instances.git",
        "glob": f"{wfname}/chameleon-cloud/*.json",
    }
    for wfname in [
        "1000genome",
        "cycles",
        "epigenomics",
        "montage",
        "seismology",
        "soykb",
        "srasearch",
    ]
}
makeflow_workflows = {
    wfname: {
        "repo": "https://github.com/wfcommons/makeflow-instances.git",
        "glob": f"{wfname}/chameleon-cloud/*.json",
    }
    for wfname in ["blast", "bwa"]
}
workflow_instances = {**pegasus_workflows, **makeflow_workflows}
workflow_instances["genome"] = workflow_instances["1000genome"]


def get_real_workflows(wfname: str) -> List[nx.DiGraph]:
    """Get workflows representing the specified wfname.

    Args:
        wfname (str): The name of the workflow.

    Returns:
        List[nx.DiGraph]: The workflows as NetworkX directed graphs.

    Raises:
        ValueError: If the wfname is not found.
    """
    if wfname not in workflow_instances:
        raise ValueError(
            f"Workflow {wfname} not found. Available workflows: {workflow_instances.keys()}"
        )

    repo = workflow_instances[wfname]["repo"]
    glob_pattern = workflow_instances[wfname]["glob"]

    path_repo = download_repo(repo)
    workflows: List[nx.DiGraph] = []
    for path in pathlib.Path(path_repo).glob(glob_pattern):
        recipe_name = path.parent.parent.name
        workflows.append(trace_to_digraph(path, recipe_name))
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
    distribution_types = ["norm", "expon", "lognorm", "gamma", "beta", "uniform"]
    fit_results = {}
    for dist_type in distribution_types:
        params = getattr(stats, dist_type).fit(data)
        D, p_value = stats.kstest(data, dist_type, args=params)
        fit_results[dist_type] = {"params": params, "D": D, "p_value": p_value}

    best_dist = min(fit_results, key=lambda k: fit_results[k]["D"])
    best_fit_params = fit_results[best_dist]["params"]

    min_data = min(data)
    max_data = max(data)

    return lambda num: [
        max(min_data, min(max_data, float(value)))
        for value in getattr(stats, best_dist).rvs(size=num, *best_fit_params)
    ]


def get_networks(
    num: int, cloud_name: str, network_speed: float = 100
) -> List[Network]:
    """Generate random networks based on real cloud configurations.

    Since Chameleon cloud uses a shared file-system for communication and network speeds are very high
    (10-100 Gbps), the communication bottleneck is the SSD speed. Default network speed is based on
    specification of SSD in Chameleon cloud at time of writing (11/19/2023).
    Source: https://www.disctech.com/Seagate-ST2000NX0273-2000GB-SAS-Hard-Drive

    Args:
        num (int): The number of networks to generate.
        cloud_name (str): The name of the cloud.
        network_speed (float, optional): The speed of the network in MegaBytes per second. Defaults to 100.

    Returns:
        List[Network]: The list of networks.
    """
    real_networks = get_real_networks(cloud_name)
    get_num_nodes = get_best_fit(
        [network.number_of_nodes() for network in real_networks]
    )
    get_node_speed = get_best_fit(
        [
            network.nodes[node]["weight"]
            for network in real_networks
            for node in network.nodes
        ]
    )

    all_num_nodes = list(map(int, get_num_nodes(num)))
    networks: List[Network] = []
    for num_nodes in all_num_nodes:
        node_speeds = get_node_speed(num_nodes)
        nodes = [(f"N{i}", max(1e-9, node_speeds[i])) for i in range(num_nodes)]
        edges = [
            (f"N{src}", f"N{dst}", network_speed if src != dst else 1e9)
            for src, dst in product(range(num_nodes), range(num_nodes))
        ]
        networks.append(Network.create(nodes=nodes, edges=edges))

    return networks


def trace_to_digraph(path: Union[str, pathlib.Path], recipe_name: str) -> nx.DiGraph:
    """Convert a WfCommons trace (v1.4 schema) to a NetworkX DiGraph.

    Args:
        path: Path to the JSON trace file.
        recipe_name: Name of the recipe (for logging purposes).

    Returns:
        nx.DiGraph: The task graph with 'weight' attributes on nodes and edges.
    """
    trace = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

    workflow = nx.DiGraph()

    # Build mapping from task name to output files
    task_outputs: Dict[str, Dict[str, float]] = {}  # task_name -> {file_name: size}
    task_name_to_id: Dict[str, str] = {}  # task name -> task id

    for task in trace["workflow"]["tasks"]:
        task_id = task.get("id", task["name"])
        task_name = task["name"]
        task_name_to_id[task_name] = task_id

        # Get runtime
        runtime = task.get("runtimeInSeconds", task.get("runtime", 1e-9))
        workflow.add_node(task_id, weight=max(1e-9, float(runtime)))

        # Track output files for this task
        task_outputs[task_name] = {}
        for file_info in task.get("files", []):
            if file_info["link"] == "output":
                task_outputs[task_name][file_info["name"]] = file_info.get(
                    "sizeInBytes", 0
                )

    # Add edges based on parent relationships
    for task in trace["workflow"]["tasks"]:
        task_id = task.get("id", task["name"])
        task_name = task["name"]

        # Get input files for this task
        input_files: Set[str] = set()
        for file_info in task.get("files", []):
            if file_info["link"] == "input":
                input_files.add(file_info["name"])

        # For each parent, compute edge weight based on matching output->input files
        for parent_name in task.get("parents", []):
            parent_id = task_name_to_id.get(parent_name, parent_name)

            # Sum up sizes of files that parent outputs and this task inputs
            edge_weight = 0.0
            parent_outputs = task_outputs.get(parent_name, {})
            for file_name, file_size in parent_outputs.items():
                if file_name in input_files:
                    edge_weight += file_size

            workflow.add_edge(parent_id, task_id, weight=max(1e-9, edge_weight))

    return workflow


def get_workflow_task_info(recipe_name: str) -> Dict:
    """Get the task type statistics for the given recipe.

    Args:
        recipe_name (str): The name of the recipe.

    Returns:
        Dict: The task type statistics.
    """
    _check_wfcommons_available()
    path = pathlib.Path(wfchef.__file__).parent.joinpath(
        "recipes", recipe_name, "task_type_stats.json"
    )
    if not path.exists():
        raise ValueError(
            f"Recipe {recipe_name} not found. Available recipes: {recipes.keys()}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def get_workflows(
    num: int, recipe_name: str, max_size_multiplier: Optional[int] = None
) -> List[TaskGraph]:
    """Generate a list of task graphs for the given recipe.

    Args:
        num (int): The number of task graphs to generate.
        recipe_name (str): The name of the recipe.
        max_size_multiplier (int, optional): Maximum size multiplier for tasks. Defaults to None.

    Returns:
        List[TaskGraph]: The list of task graphs.
    """
    _check_wfcommons_available()
    if recipe_name not in recipes:
        raise ValueError(
            f"Recipe {recipe_name} not found. Available recipes: {recipes.keys()}"
        )

    task_graphs: List[TaskGraph] = []
    for _ in range(num):
        min_tasks, max_tasks = get_num_task_range(recipe_name)
        if max_size_multiplier is not None:
            max_tasks = max_size_multiplier * min_tasks
        num_tasks = random.randint(min_tasks, max_tasks)
        recipe = recipes[recipe_name](num_tasks=num_tasks)  # type: ignore
        generator = WorkflowGenerator(recipe)
        workflow = generator.build_workflow()
        with tempfile.NamedTemporaryFile() as tmp:
            workflow.write_json(pathlib.Path(tmp.name))
            digraph = trace_to_digraph(tmp.name, recipe_name)

            # Add source and sink nodes if needed to ensure single source/sink
            sources = [n for n in digraph.nodes if digraph.in_degree(n) == 0]
            sinks = [n for n in digraph.nodes if digraph.out_degree(n) == 0]

            if len(sources) > 1:
                digraph.add_node("SRC", weight=1e-9)
                for src in sources:
                    digraph.add_edge("SRC", src, weight=1e-9)

            if len(sinks) > 1:
                digraph.add_node("DST", weight=1e-9)
                for sink in sinks:
                    digraph.add_edge(sink, "DST", weight=1e-9)

            task_graphs.append(TaskGraph.from_nx(digraph))

    return task_graphs


def get_wfcommons_instance(
    recipe_name: str,
    ccr: float,
    estimate_method: Callable[[RandomVariable, bool], float] = lambda x,
    is_speed: x.mean(),
    max_size_multiplier: int = 2,
) -> Tuple[Network, TaskGraph]:
    """Generate a network and workflow instance from wfcommons.

    Args:
        recipe_name (str): The name of the recipe.
        ccr (float): The desired communication-to-computation ratio.
        estimate_method (Callable[[RandomVariable, bool], float], optional): The method to estimate
            the weight from a random variable. Defaults to mean().
        max_size_multiplier (int, optional): Maximum size multiplier for tasks. Defaults to 2.
    Returns:
        Tuple[Network, TaskGraph]: The network and workflow instance.
    """
    workflow = get_workflows(
        num=1, recipe_name=recipe_name, max_size_multiplier=max_size_multiplier
    )[0].graph

    # rename weight attribute to weight_rv
    weight_rv: RandomVariable
    for node in workflow.nodes:
        weight_rv = workflow.nodes[node]["weight"]
        workflow.nodes[node]["weight_rv"] = weight_rv
        workflow.nodes[node]["weight_estimate"] = max(
            1e-9, estimate_method(weight_rv, True)
        )
        workflow.nodes[node]["weight_actual"] = max(1e-9, weight_rv.sample())
        workflow.nodes[node]["weight"] = workflow.nodes[node]["weight_estimate"]

    for u, v in workflow.edges:
        weight_rv = workflow.edges[u, v]["weight"]
        workflow.edges[u, v]["weight_rv"] = weight_rv
        workflow.edges[u, v]["weight_estimate"] = max(
            1e-9, estimate_method(weight_rv, True)
        )
        workflow.edges[u, v]["weight_actual"] = max(1e-9, weight_rv.sample())
        workflow.edges[u, v]["weight"] = workflow.edges[u, v]["weight_estimate"]

    # add src and dst task
    workflow.add_node(
        "SRC",
        weight=1e-9,
        weight_estimate=1e-9,
        weight_actual=1e-9,
        weight_rv=RandomVariable(samples=[1e-9]),
    )
    workflow.add_node(
        "DST",
        weight=1e-9,
        weight_estimate=1e-9,
        weight_actual=1e-9,
        weight_rv=RandomVariable(samples=[1e-9]),
    )
    for node in workflow.nodes:
        if node not in ["SRC", "DST"] and not workflow.in_degree(node):
            workflow.add_edge(
                "SRC",
                node,
                weight=1e9,
                weight_estimate=1e9,
                weight_actual=1e9,
                weight_rv=RandomVariable(samples=[1e9]),
            )
    for node in workflow.nodes:
        if node not in ["SRC", "DST"] and not workflow.out_degree(node):
            workflow.add_edge(
                node,
                "DST",
                weight=1e9,
                weight_estimate=1e9,
                weight_actual=1e9,
                weight_rv=RandomVariable(samples=[1e9]),
            )

    network: nx.Graph = get_networks(num=1, cloud_name="chameleon", network_speed=1.0)[
        0
    ].graph

    for node in network.nodes:
        weight_rv = network.nodes[node]["weight"]
        network.nodes[node]["weight_rv"] = weight_rv
        network.nodes[node]["weight_estimate"] = max(
            1e-9,
            estimate_method(weight_rv, True)
            if isinstance(weight_rv, RandomVariable)
            else weight_rv,
        )
        network.nodes[node]["weight_actual"] = max(
            1e-9,
            weight_rv.sample() if isinstance(weight_rv, RandomVariable) else weight_rv,
        )
        network.nodes[node]["weight"] = network.nodes[node]["weight_estimate"]

    # adjust network edges to match CCR
    avg_task_cost = np.mean(
        [workflow.nodes[node]["weight_actual"] for node in workflow.nodes]
    )
    avg_dep_cost = np.mean(
        [workflow.edges[u, v]["weight_actual"] for u, v in workflow.edges]
    )
    avg_node_speed = np.mean(
        [network.nodes[node]["weight_actual"] for node in network.nodes]
    )
    avg_comm_speed = float(ccr * avg_dep_cost / (avg_task_cost / avg_node_speed))
    for u, v in network.edges:
        weight_rv = RandomVariable(samples=[avg_comm_speed] * 100)
        network.edges[u, v]["weight_rv"] = weight_rv
        network.edges[u, v]["weight_estimate"] = max(
            1e-9,
            estimate_method(weight_rv, True)
            if isinstance(weight_rv, RandomVariable)
            else weight_rv,
        )
        network.edges[u, v]["weight_actual"] = max(
            1e-9,
            weight_rv.sample() if isinstance(weight_rv, RandomVariable) else weight_rv,
        )
        network.edges[u, v]["weight"] = network.edges[u, v]["weight_estimate"]

    return Network.from_nx(network), TaskGraph.from_nx(workflow)
