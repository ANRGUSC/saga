"""Coupled compute/network simulation with link contention.

SAGA's core timing model gives every transfer the full link rate:

    arrival = parent.end + edge.size / link.speed

so any number of concurrent transfers over a link each run at full speed,
and a node runs one task at a time. Both assumptions are optimistic on a
real cluster: a fan-in delivers several flows into one NIC at once, and
Kubernetes runs co-located pods concurrently while CPU allows.

This module simulates a *placement* under two coupled resource models and
returns two schedules — one for compute, one for the network — so they can
be inspected, plotted and compared separately.

Network model (TCP-like fair share)
-----------------------------------
Every transfer is a flow that contends for three resources: the sending
node's NIC, the link itself, and the receiving node's NIC. A flow's rate is

    rate(f) = min over resources r used by f of  capacity(r) / flows_on(r)

i.e. each resource splits its capacity equally among the flows crossing it,
and a flow runs at its most constrained resource. Rates are recomputed at
every event (a flow finishing, a task finishing and releasing new flows),
so a flow speeds up when a competitor completes — the behaviour TCP
approximates. This is a fluid model: no slow start, no RTT effects, no
packet-level dynamics.

Modelling both NICs matters for DAG shapes that schedulers actually
produce. Per-link sharing alone would miss fan-in entirely: four producers
on four different nodes sending to one consumer use four *different* links,
so without an ingress resource they would all run at full rate — exactly
the case a scheduler most needs to get right.

Compute model
-------------
A node is a capacity, not a lock. A task starts when its inputs have
arrived and the node has room for its CPU request; several tasks may run
concurrently. Passing ``exclusive=True`` recovers SAGA's one-task-at-a-time
behaviour, which makes the cost of that assumption measurable.
"""

from __future__ import annotations


from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from saga import Network, ScheduledTask, Schedule, TaskGraph

__all__ = ["NetworkTransfer", "SimulationResult", "simulate_placement"]

_EPS = 1e-9
_SUPER_NODES = ("__super_source__", "__super_sink__")


class NetworkTransfer(BaseModel):
    """One DAG edge moving between two nodes, as scheduled on the network."""

    src_task: str
    dst_task: str
    src_node: str
    dst_node: str
    size: float
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def link(self) -> Tuple[str, str]:
        return (self.src_node, self.dst_node)

    @property
    def mean_rate(self) -> float:
        """Achieved rate: size / duration. Below the link speed whenever the
        flow shared a resource with another flow."""
        return self.size / self.duration if self.duration > 0 else float("inf")


class SimulationResult(BaseModel):
    """Compute and network schedules for one placement."""

    tasks: Dict[str, ScheduledTask]
    transfers: List[NetworkTransfer]

    @property
    def makespan(self) -> float:
        return max((t.end for t in self.tasks.values()), default=0.0)

    def by_node(self) -> Dict[str, List[ScheduledTask]]:
        out: Dict[str, List[ScheduledTask]] = {}
        for t in self.tasks.values():
            out.setdefault(t.node, []).append(t)
        for v in out.values():
            v.sort(key=lambda t: t.start)
        return out

    def by_link(self) -> Dict[Tuple[str, str], List[NetworkTransfer]]:
        out: Dict[Tuple[str, str], List[NetworkTransfer]] = {}
        for f in self.transfers:
            out.setdefault(f.link, []).append(f)
        for v in out.values():
            v.sort(key=lambda f: f.start)
        return out

    def to_schedule(self, task_graph: TaskGraph, network: Network) -> Schedule:
        """Wrap the compute half as a plain saga.Schedule."""
        sched = Schedule(task_graph, network)
        for t in sorted(self.tasks.values(), key=lambda x: x.start):
            sched.add_task(t)
        return sched


class _Flow:
    __slots__ = ("src_task", "dst_task", "src_node", "dst_node", "size",
                 "remaining", "start", "resources")

    def __init__(self, src_task, dst_task, src_node, dst_node, size, start):
        self.src_task = src_task
        self.dst_task = dst_task
        self.src_node = src_node
        self.dst_node = dst_node
        self.size = size
        self.remaining = size
        self.start = start
        # A flow contends at the sender's NIC, the link, and the receiver's NIC.
        self.resources = (("egress", src_node),
                          ("link", src_node, dst_node),
                          ("ingress", dst_node))


def _fair_share_rates(flows: List[_Flow], capacity) -> Dict[int, float]:
    """Equal split per resource; each flow runs at its tightest resource."""
    counts: Dict[tuple, int] = {}
    for f in flows:
        for r in f.resources:
            counts[r] = counts.get(r, 0) + 1
    return {id(f): min(capacity(r) / counts[r] for r in f.resources) for f in flows}


def simulate_placement(
    network: Network,
    task_graph: TaskGraph,
    placement: Dict[str, str],
    *,
    task_cpu: Optional[Dict[str, float]] = None,
    node_cpu: Optional[Dict[str, float]] = None,
    task_duration=None,
    nic_speed: Optional[Dict[str, float]] = None,
    exclusive: bool = False,
) -> SimulationResult:
    """Time ``placement`` under contention.

    Args:
        placement: task name -> node name. Every task must be placed.
        task_cpu: task -> CPU request. Omitted tasks request nothing.
        node_cpu: node -> CPU capacity. Omitted nodes are unconstrained.
        task_duration: callable (task, node) -> seconds. Defaults to SAGA's
            ``cost / speed``; pass measured runtimes to time a real cluster.
        nic_speed: node -> NIC rate. Defaults to the fastest link the node
            has, i.e. the NIC only binds when several flows share it.
        exclusive: run one task per node at a time (SAGA's assumption).

    Returns:
        SimulationResult with a compute schedule and a network schedule.
    """
    # TaskGraph.create injects __super_source__/__super_sink__ when a graph has
    # several sources or sinks. They are normalisation artifacts with zero cost
    # and zero-size edges, so place them anywhere and strip them from results
    # rather than making every caller special-case them.
    placement = dict(placement)
    synthetic = {t.name for t in task_graph.tasks
                 if t.name in _SUPER_NODES and t.name not in placement}
    any_node = next(iter(n.name for n in network.nodes))
    for name in synthetic:
        placement[name] = any_node

    missing = [t.name for t in task_graph.tasks if t.name not in placement]
    if missing:
        raise ValueError(f"tasks not placed: {sorted(missing)}")

    node_speed = {n.name: n.speed for n in network.nodes}
    if task_duration is None:
        def task_duration(task: str, node: str) -> float:  # noqa: D401
            return task_graph.get_task(task).cost / node_speed[node]
    _user_duration = task_duration

    def task_duration(task: str, node: str) -> float:  # noqa: F811
        return 0.0 if task in synthetic else _user_duration(task, node)

    link_speed: Dict[Tuple[str, str], float] = {}
    for u in node_speed:
        for v in node_speed:
            if u != v:
                link_speed[(u, v)] = network.get_edge(u, v).speed

    if nic_speed is None:
        nic_speed = {
            n: max((link_speed[(n, m)] for m in node_speed if m != n), default=float("inf"))
            for n in node_speed
        }

    def capacity(resource) -> float:
        kind = resource[0]
        if kind == "link":
            return link_speed[(resource[1], resource[2])]
        return nic_speed[resource[1]]

    cpu_need = dict(task_cpu or {})
    cpu_cap = dict(node_cpu or {})

    deps: Dict[str, List] = {t.name: list(task_graph.in_edges(t.name)) for t in task_graph.tasks}
    unscheduled = set(placement)
    done: Dict[str, float] = {}                 # task -> end time
    arrived: Dict[Tuple[str, str], float] = {}  # (dep, consumer) -> arrival time
    running: List[Tuple[float, str, str, float]] = []  # (end, task, node, cpu)
    flows: List[_Flow] = []
    result_tasks: Dict[str, ScheduledTask] = {}
    transfers: List[NetworkTransfer] = []

    now = 0.0
    guard = 0
    while unscheduled or running or flows:
        guard += 1
        if guard > 100_000:
            raise RuntimeError("simulation did not converge")

        # --- start every task whose inputs have arrived and that fits -------
        progressed = True
        while progressed:
            progressed = False
            for name in sorted(unscheduled):
                node = placement[name]
                ready = 0.0
                blocked = False
                for e in deps[name]:
                    key = (e.source, name)
                    if key not in arrived:
                        blocked = True
                        break
                    ready = max(ready, arrived[key])
                if blocked or ready > now + _EPS:
                    continue
                if exclusive:
                    fits = not any(n == node for _, _, n, _ in running)
                else:
                    used = sum(c for _, _, n, c in running if n == node)
                    fits = cpu_cap.get(node) is None or \
                        used + cpu_need.get(name, 0.0) <= cpu_cap[node] + _EPS
                if not fits:
                    continue
                end = now + task_duration(name, node)
                result_tasks[name] = ScheduledTask(node=node, name=name, start=now, end=end)
                running.append((end, name, node, cpu_need.get(name, 0.0)))
                unscheduled.discard(name)
                progressed = True

        if not (running or flows):
            if unscheduled:
                raise RuntimeError(f"deadlock with tasks pending: {sorted(unscheduled)}")
            break

        # --- advance to the next event -------------------------------------
        rates = _fair_share_rates(flows, capacity) if flows else {}
        t_task = min((e for e, _, _, _ in running), default=float("inf"))
        t_flow = float("inf")
        for f in flows:
            r = rates[id(f)]
            t_flow = min(t_flow, now + (f.remaining / r if r > 0 else float("inf")))
        step_to = min(t_task, t_flow)
        if step_to == float("inf"):
            raise RuntimeError("no progress possible (zero-capacity resource?)")

        dt = step_to - now
        for f in flows:
            f.remaining = max(0.0, f.remaining - rates[id(f)] * dt)
        now = step_to

        # --- retire finished flows ------------------------------------------
        still: List[_Flow] = []
        for f in flows:
            if f.remaining <= _EPS:
                transfers.append(NetworkTransfer(
                    src_task=f.src_task, dst_task=f.dst_task,
                    src_node=f.src_node, dst_node=f.dst_node,
                    size=f.size, start=f.start, end=now))
                arrived[(f.src_task, f.dst_task)] = now
            else:
                still.append(f)
        flows = still

        # --- retire finished tasks, release their outgoing edges ------------
        finished = [r for r in running if r[0] <= now + _EPS]
        running = [r for r in running if r[0] > now + _EPS]
        for end, name, node, _ in finished:
            done[name] = end
            for e in task_graph.out_edges(name):
                consumer = e.target
                if placement[consumer] == node:
                    arrived[(name, consumer)] = end        # same node: no transfer
                elif e.size <= 0:
                    arrived[(name, consumer)] = end        # nothing to move
                else:
                    flows.append(_Flow(name, consumer, node, placement[consumer],
                                       e.size, end))

    for name in synthetic:
        result_tasks.pop(name, None)
    return SimulationResult(tasks=result_tasks, transfers=transfers)
