import json
import pathlib
import random
import tempfile
from itertools import product
from typing import Dict, List, Optional, Set, Union

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
                f"N{i}",  # StochasticNetwork requires string node names
                weight=RandomVariable(
                    samples=np.clip(
                        get_node_speed(DEFAULT_NUM_SAMPLES), 1e-9, 1e9
                    ).tolist()
                ),
            )

        for src, dst in product(network.nodes, network.nodes):
            # from_nx expects RandomVariable weights, so wrap the constant link speed.
            speed = network_speed if src != dst else 1e9
            network.add_edge(src, dst, weight=RandomVariable(samples=[speed]))

        networks.append(network)

    return networks


def _sample_rv(dist_info: Dict) -> Union[RandomVariable, None]:
    """Draw a RandomVariable from a fitted distribution entry.

    Expects a dict shaped like {"distribution": {"name", "params"}, "min", "max"}.
    Returns None when no distribution is present.
    """
    dist = dist_info.get("distribution") if dist_info else None
    if not dist:
        return None
    samples = getattr(stats, dist["name"]).rvs(
        *dist["params"], size=DEFAULT_NUM_SAMPLES
    )
    samples = np.clip(samples, dist_info["min"], dist_info["max"])
    return RandomVariable(samples=samples.tolist())


def _trace_to_digraph_v15(trace: Dict, task_type_info: Dict) -> nx.DiGraph:
    """Build a stochastic task graph from a WfCommons v1.5 trace.

    Task runtimes and output-file sizes are drawn as RandomVariables from the recipe's
    fitted per-task-type distributions (task_type_stats.json). Each task's type is its
    specification name; edge weights sum the sizes (in MB) of files a parent outputs and
    the child inputs. Falls back to the trace's own scalar sizes when a distribution is
    missing.
    """
    spec = trace["workflow"]["specification"]
    spec_tasks = spec["tasks"]
    real_size = {
        f["id"]: float(f.get("sizeInBytes", 0.0)) for f in spec.get("files", [])
    }
    task_ids: Set[str] = {t["id"] for t in spec_tasks}

    task_graph = nx.DiGraph()
    out_files: Dict[str, Dict[str, RandomVariable]] = {}
    in_files: Dict[str, Set[str]] = {}

    for task in spec_tasks:
        task_id, task_type = task["id"], task["name"]
        info = task_type_info.get(task_type, {})
        runtime = _sample_rv(info.get("runtime", {})) or RandomVariable(samples=[1e-9])
        task_graph.add_node(task_id, weight=runtime)

        out_files[task_id] = {}
        for file_id in task.get("outputFiles", []):
            file_type = file_id[36:]  # strip the 36-char uuid, leaving the extension
            size_rv = _sample_rv((info.get("output") or {}).get(file_type, {}))
            if size_rv is None:
                size_rv = RandomVariable(samples=[real_size.get(file_id, 1e-9)])
            out_files[task_id][file_id] = size_rv * (1.0 / 1e6)  # bytes to MB
        in_files[task_id] = set(task.get("inputFiles", []))

    for task in spec_tasks:
        child_id = task["id"]
        for parent_id in task.get("parents", []):
            if parent_id not in task_ids:
                continue
            matched = [
                rv
                for file_id, rv in out_files.get(parent_id, {}).items()
                if file_id in in_files[child_id]
            ]
            weight = (
                sum(matched[1:], matched[0])
                if matched
                else RandomVariable(samples=[1e-9])
            )
            task_graph.add_edge(parent_id, child_id, weight=weight)

    return task_graph


def _trace_to_digraph_v14(trace: Dict, task_type_info: Dict) -> nx.DiGraph:
    """Build a stochastic task graph from a WfCommons v1.4 (Pegasus) trace."""
    for task in trace["workflow"]["tasks"]:
        task_type = task["command"]["program"]
        task_info = task_type_info[task_type]
        task["runtime"] = _sample_rv(task_info.get("runtime", {})) or RandomVariable(
            samples=[1e-9]
        )

        for io_file in task["files"]:
            file_type = io_file["name"][36:]  # strip off uuid
            size_rv = _sample_rv(task_info[io_file["link"]].get(file_type, {}))
            if size_rv is not None:
                io_file["size"] = size_rv

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
                f"Task {task['name']} has no 'runtime' or 'runtimeInSeconds' attribute. {list(task.keys())}."
            )
        task_graph.add_node(task["name"], weight=runtime)

        for io_file in task["files"]:
            if io_file["link"] == "output":
                size_in_bytes = io_file.get("sizeInBytes", io_file.get("size"))
                if size_in_bytes is None:
                    raise ValueError(
                        f"File {io_file['name']} has no 'sizeInBytes' or 'size' attribute."
                    )
                outputs.setdefault(task["name"], {})[io_file["name"]] = (
                    size_in_bytes / 1e6
                )
            elif io_file["link"] == "input":
                input_files.setdefault(task["name"], set()).add(io_file["name"])

    for task in trace["workflow"]["tasks"]:
        for child in task.get("children", []):
            weight = 0.0
            for input_file in input_files.get(child, []):
                weight += outputs[task["name"]].get(input_file, 0)
            task_graph.add_edge(task["name"], child, weight=weight)

    return task_graph


def trace_to_digraph(
    path: Union[str, pathlib.Path], task_type_info: Dict
) -> nx.DiGraph:
    """Convert a WfCommons trace to a stochastic networkx DiGraph.

    Supports both the v1.5 (specification/execution) and v1.4 (Pegasus) schemas.

    Args:
        path (Union[str, pathlib.Path]): The path to the trace.
        task_type_info (Dict): The fitted per-task-type distributions.

    Returns:
        nx.DiGraph: The task graph with RandomVariable weights on nodes and edges.
    """
    trace = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if str(trace.get("schemaVersion", "1.4")).startswith("1.5"):
        return _trace_to_digraph_v15(trace, task_type_info)
    return _trace_to_digraph_v14(trace, task_type_info)


def get_workflows(
    num: int, recipe_name: str, size_cap: Optional[int] = None
) -> List[nx.DiGraph]:
    """Generate a list of network, task graph pairs for the given recipe.

    Args:
        num (int): The number of task graphs to generate.
        recipe_name (str): The name of the recipe.
        size_cap (int, optional): Absolute cap on the number of tasks per workflow. When
            set, sizes are drawn from [min_tasks, min(max_tasks, size_cap)]. Defaults to None.

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
        min_tasks, max_tasks = get_num_task_range(recipe_name)
        if size_cap is not None:
            max_tasks = min(max_tasks, size_cap)
        min_tasks = min(min_tasks, max_tasks)
        num_tasks = random.randint(min_tasks, max_tasks)
        recipe = recipes[recipe_name](num_tasks=num_tasks)  # type: ignore
        generator = WorkflowGenerator(recipe)  # type: ignore[misc]  # None only if wfcommons is not installed
        workflow = generator.build_workflow()
        with tempfile.NamedTemporaryFile() as tmp:
            workflow.write_json(pathlib.Path(tmp.name))
            task_graphs.append(trace_to_digraph(tmp.name, task_type_info))

    return task_graphs
