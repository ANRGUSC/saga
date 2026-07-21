"""Problem-instance builders for the two experiment branches (see EXPERIMENT_PLAN.md).

Each builder returns a list of Instance(name, network, task_graph, node_constraints),
either deterministic or stochastic depending on `regime`. Base instances are built once
per (workflow, regime); `scaled` produces the per-CCR copies so a workflow is generated
only once and then rescaled across the CCR sweep.

- RIoTBench branch: tiered edge/fog/cloud dataflows with per-task placement constraints
  (sources and sinks pinned to the edge, storage and training to the cloud). Without the
  constraints the fast cloud node swallows every task and placement is trivial.
- WfCommons branch: scientific workflows on a scaled network, no constraints.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np

from saga.stochastic import StochasticNetwork, StochasticTaskGraph
from saga.schedulers.data import riotbench as det_riot
from saga.schedulers.stochastic.data import riotbench as sto_riot
from saga.schedulers.data.wfcommons import get_workflows as det_wf, get_networks as det_net
from saga.schedulers.stochastic.data.wfcommons import (
    get_workflows as sto_wf,
    get_networks as sto_net,
)


@dataclass
class Instance:
    name: str
    network: Any        # Network or StochasticNetwork
    task_graph: Any     # TaskGraph or StochasticTaskGraph
    node_constraints: Optional[Dict[str, Set[str]]] = None


def scaled(instance: Instance, ccr: float) -> Instance:
    """Return a copy of `instance` with its network scaled to the target CCR."""
    return Instance(
        name=instance.name,
        network=instance.network.scale_to_ccr(instance.task_graph, ccr),
        task_graph=instance.task_graph,
        node_constraints=instance.node_constraints,
    )


# ---------------------------------------------------------------------------
# RIoTBench branch
# ---------------------------------------------------------------------------
RIOTBENCH_DATAFLOWS = ("etl", "stats", "predict", "train")

_DET_RIOT = {
    "etl": det_riot.get_etl_task_graphs,
    "stats": det_riot.get_stats_task_graphs,
    "predict": det_riot.get_predict_task_graphs,
    "train": det_riot.get_train_task_graphs,
}
_STO_RIOT = {
    "etl": sto_riot.get_etl_task_graphs,
    "stats": sto_riot.get_stats_task_graphs,
    "predict": sto_riot.get_predict_task_graphs,
    "train": sto_riot.get_train_task_graphs,
}

# Tier assignment: operators that must live at the edge (sensors/actuators) or in the
# cloud (storage and model training). Everything else is free to run on any tier.
EDGE_OPS = {"Source", "TimerSource", "VirtualSource", "Sink"}
CLOUD_OPS = {
    "TableRead", "AzureTableInsert", "BlobUpload", "BlobWrite", "BlobRead",
    "DecisionTreeTrain", "MultiVarLinearRegTrain",
}


def _tier_constraints(task_names, node_names) -> Dict[str, Set[str]]:
    edge = {n for n in node_names if n.startswith("E")}
    cloud = {n for n in node_names if n.startswith("C")}
    cons: Dict[str, Set[str]] = {}
    for name in task_names:
        if name in EDGE_OPS:
            cons[name] = set(edge)
        elif name in CLOUD_OPS:
            cons[name] = set(cloud)
    return cons


def riotbench_base(
    dataflow: str,
    n: int,
    regime: str,
    *,
    seed: int = 0,
    cv: float = 0.3,
    num_edge: int = 2,
    num_fog: int = 2,
    num_cloud: int = 1,
) -> List[Instance]:
    """Build `n` RIoTBench instances (unscaled) for one dataflow and regime."""
    np.random.seed(seed)
    net_kwargs = dict(num_edges_nodes=num_edge, num_fog_nodes=num_fog, num_cloud_nodes=num_cloud)
    if regime == "stochastic":
        networks = sto_riot.get_fog_networks(n, cv=cv, **net_kwargs)
        task_graphs = _STO_RIOT[dataflow](n, cv=cv)
    else:
        networks = det_riot.get_fog_networks(n, **net_kwargs)
        task_graphs = _DET_RIOT[dataflow](n)

    instances = []
    for i in range(n):
        net, tg = networks[i], task_graphs[i]
        constraints = _tier_constraints(
            [t.name for t in tg.tasks], [node.name for node in net.nodes]
        )
        instances.append(Instance(f"{dataflow}_{i}", net, tg, constraints))
    return instances


# ---------------------------------------------------------------------------
# WfCommons branch
# ---------------------------------------------------------------------------
WFCOMMONS_WORKFLOWS = (
    "epigenomics", "montage", "cycles", "seismology",
    "soykb", "srasearch", "genome", "blast", "bwa",
)

# Cap workflow size so the large recipes (epigenomics, montage, etc.) stay tractable,
# especially for the stochastic simulation. All recipe minimums are below this.
MAX_TASKS = 300


def wfcommons_base(
    workflow: str,
    n: int,
    regime: str,
    *,
    seed: int = 0,
    cloud: str = "chameleon",
) -> List[Instance]:
    """Build `n` WfCommons instances (unscaled) for one workflow and regime."""
    np.random.seed(seed)
    if regime == "stochastic":
        task_graphs = [StochasticTaskGraph.from_nx(g) for g in sto_wf(n, workflow, size_cap=MAX_TASKS)]
        networks = [StochasticNetwork.from_nx(g) for g in sto_net(n, cloud)]
    else:
        task_graphs = det_wf(n, workflow, size_cap=MAX_TASKS)
        networks = det_net(n, cloud)
    return [Instance(f"{workflow}_{i}", networks[i], task_graphs[i]) for i in range(n)]


def base_instances(branch: str, workflow: str, n: int, regime: str, *, seed: int = 0) -> List[Instance]:
    """Dispatch to the right branch builder."""
    if branch == "riotbench":
        return riotbench_base(workflow, n, regime, seed=seed)
    if branch == "wfcommons":
        return wfcommons_base(workflow, n, regime, seed=seed)
    raise ValueError(f"Unknown branch: {branch}")


def workflows_for(branch: str) -> tuple:
    return RIOTBENCH_DATAFLOWS if branch == "riotbench" else WFCOMMONS_WORKFLOWS
