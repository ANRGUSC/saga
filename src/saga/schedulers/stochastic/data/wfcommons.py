import json
import pathlib
import random
import tempfile
from itertools import product
from typing import Dict, List, Set, Union

import networkx as nx
import numpy as np
from scipy import stats

from saga.utils.random_variable import RandomVariable, DEFAULT_NUM_SAMPLES
from saga.schedulers.data.wfcommons import (
    recipes,
    get_num_task_range,
    get_real_networks,
    get_best_fit,
    get_workflow_task_info,
    WFCOMMONS_AVAILABLE,
    _check_wfcommons_available,
)

# wfcommons is an optional dependency (does not install on Windows)
if WFCOMMONS_AVAILABLE:
    from wfcommons.wfgen.generator import WorkflowGenerator
else:
    WorkflowGenerator = None  # type: ignore


def get_networks(
    num: int, cloud_name: str, network_speed: float = 100
) -> List[nx.Graph]:
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
    networks: List[nx.Graph] = []
    for num_nodes in all_num_nodes:
        network = nx.Graph()
        for i in range(num_nodes):
            network.add_node(
                i,
                weight=RandomVariable(
                    samples=np.clip(
                        get_node_speed(DEFAULT_NUM_SAMPLES), 1e-9, 1e9
                    ).tolist()
                ),
            )

        for src, dst in product(network.nodes, network.nodes):
            network.add_edge(src, dst, weight=network_speed if src != dst else 1e9)

        networks.append(network)

    return networks


def trace_to_digraph(
    path: Union[str, pathlib.Path], task_type_info: Dict
) -> nx.DiGraph:
    """Convert a trace to a networkx DiGraph.

    Args:
        path (Union[str, pathlib.Path]): The path to the trace.
        task_type_info (Dict): The task type statistics.

    Returns:
        nx.DiGraph: The networkx DiGraph.
    """
    trace = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    # convert trace to stochastic using distribution info in task_type_info
    for task in trace["workflow"]["tasks"]:
        task_type = task["command"]["program"]
        task_info = task_type_info[task_type]

        params = task_info["runtime"]["distribution"]["params"]
        samples = getattr(stats, task_info["runtime"]["distribution"]["name"]).rvs(
            size=DEFAULT_NUM_SAMPLES, *params
        )
        samples = np.clip(
            samples, task_info["runtime"]["min"], task_info["runtime"]["max"]
        )
        task["runtime"] = RandomVariable(samples=samples)

        for io_file in task["files"]:
            link_type = io_file["link"]
            file_name = io_file["name"]
            file_type = file_name[36:]  # strip off uuid
            io_info = task_info[link_type].get(file_type)
            if io_info and io_info["distribution"]:
                samples = getattr(stats, io_info["distribution"]["name"]).rvs(
                    size=DEFAULT_NUM_SAMPLES, *io_info["distribution"]["params"]
                )
                samples = np.clip(samples, io_info["min"], io_info["max"])
                io_file["size"] = RandomVariable(samples=samples)
            else:
                pass  # leave size as float

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

        runtime = task.get("runtime", task.get("runtimeInSeconds"))
        if runtime is None:
            raise ValueError(
                f"Task {task['name']} has no 'runtime' or 'runtimeInSeconds' attribute. {list(task.keys())}. {path}"
            )
        task_graph.add_node(task["name"], weight=max(1e-9, runtime))

        for io_file in task["files"]:
            if io_file["link"] == "output":
                size_in_bytes = io_file.get("sizeInBytes", io_file.get("size"))
                if size_in_bytes is None:
                    raise ValueError(
                        f"File {io_file['name']} has no 'sizeInBytes' or 'size' attribute. {path}"
                    )
                outputs.setdefault(task["name"], {})[io_file["name"]] = (
                    size_in_bytes / 1e6
                )  # convert to MB
            elif io_file["link"] == "input":
                input_files.setdefault(task["name"], set()).add(io_file["name"])

    for task in trace["workflow"]["tasks"]:
        for child in task.get("children", []):
            weight = 0.0
            for input_file in input_files.get(child, []):
                weight += outputs[task["name"]].get(input_file, 0)
            task_graph.add_edge(task["name"], child, weight=max(1e-9, weight))

    return task_graph


def get_workflows(num: int, recipe_name: str) -> List[nx.DiGraph]:
    """Generate a list of network, task graph pairs for the given recipe.

    Args:
        num (int): The number of task graphs to generate.
        recipe_name (str): The name of the recipe.

    Returns:
        List[nx.DiGraph]: The list of task graphs.
    """
    _check_wfcommons_available()
    if recipe_name not in recipes:
        raise ValueError(
            f"Recipe {recipe_name} not found. Available recipes: {recipes.keys()}"
        )

    task_type_info = get_workflow_task_info(recipe_name)
    task_graphs: List[nx.DiGraph] = []
    for _ in range(num):
        num_tasks = random.randint(*get_num_task_range(recipe_name))
        recipe = recipes[recipe_name](num_tasks=num_tasks)  # type: ignore
        generator = WorkflowGenerator(recipe)
        workflow = generator.build_workflow()
        with tempfile.NamedTemporaryFile() as tmp:
            workflow.write_json(pathlib.Path(tmp.name))
            task_graphs.append(trace_to_digraph(tmp.name, task_type_info))

    return task_graphs
