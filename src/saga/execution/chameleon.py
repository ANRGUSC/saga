"""Helpers for building a SAGA ``Network`` from a Chameleon Cloud lease.

Chameleon (https://www.chameleoncloud.org) is a bare-metal testbed that
publishes the hardware spec of every node in its catalog (`hardware.json`
per site).  The functions in this module turn a small Python description of
a reservation — hostnames plus optional measured bandwidth — into a SAGA
``Network`` ready to be paired with any task graph.

The actual "reserve a lease and provision workers" flow lives in
``scripts/examples/dagprofiler_chameleon/provision_workers.sh``: this module
intentionally does not depend on ``python-chi`` because the rest of SAGA is
CPU / metadata only.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

from saga import Network


@dataclass
class ChameleonNode:
    """One Chameleon Cloud compute node.

    Attributes:
        name: Hostname as returned by ``openstack server list``
            (e.g. ``saga-worker-0``).  This is also the name SAGA uses to
            refer to the node and the Work Queue feature it advertises.
        speed: Compute speed in SAGA's units.  The most portable
            convention is GFLOPS or "normalized-HEFT work-units/second"; a
            simple baseline is ``cpu_cores * base_clock_ghz``.
    """

    name: str
    speed: float


def build_network(
    nodes: Iterable[ChameleonNode],
    *,
    default_bandwidth: float = 1.25e8,  # 1 Gbps in bytes/s
    bandwidths: Optional[Dict[Tuple[str, str], float]] = None,
) -> Network:
    """Build a SAGA ``Network`` from a list of Chameleon nodes.

    Args:
        nodes: Iterable of ``ChameleonNode`` instances.
        default_bandwidth: Inter-node bandwidth in bytes/second to use for
            any edge not listed in ``bandwidths``.  The default of
            ``1.25e8`` (≈1 Gbps) matches Chameleon's default management
            network; if you've reserved a dedicated 25/100 Gbps fabric,
            override it.
        bandwidths: Optional per-edge overrides, keyed by unordered
            ``(name_a, name_b)`` tuples.  The ``Network.create`` helper
            handles symmetrisation.

    Returns:
        A fully connected SAGA ``Network``.
    """
    nodes = list(nodes)
    if not nodes:
        raise ValueError("Chameleon network must have at least one node")

    overrides = {tuple(sorted(k)): float(v) for k, v in (bandwidths or {}).items()}

    node_specs: List[Tuple[str, float]] = [(n.name, float(n.speed)) for n in nodes]
    edge_specs: List[Tuple[str, str, float]] = []
    for a, b in combinations(nodes, 2):
        key = tuple(sorted((a.name, b.name)))
        edge_specs.append((a.name, b.name, overrides.get(key, default_bandwidth)))

    return Network.create(nodes=node_specs, edges=edge_specs)
