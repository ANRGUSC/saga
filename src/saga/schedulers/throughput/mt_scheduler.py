"""MT-Scheduler: Maximum Throughput Scheduler for heterogeneous DAG scheduling.

Adapted from:
    Al-Sinayyid, A., & Zhu, M. (2020). "Job scheduler for streaming applications
    in heterogeneous distributed processing systems."
    Journal of Supercomputing, 76(12), 9609-9628.
    https://doi.org/10.1007/s11227-020-03223-z

The original paper targets Apache Storm streaming topologies. We adapt it to
static DAG scheduling by interpreting throughput as 1 / bottleneck_stage_time
for pipelined execution. The algorithm itself is unchanged: linearize the DAG
via topological sort, identify the critical path, then use dynamic programming
to map critical-path tasks to nodes such that the bottleneck (max over compute
+ transfer time per stage) is minimized. Non-critical-path tasks are placed
greedily layer-by-layer.

Per the paper's own simplification (p. 9616), the critical path is identified
under a homogeneous-network assumption; the heterogeneity-aware mapping then
operates on that fixed CP. The metadata grouping / site tag feature is omitted
(the paper notes that without it, the cluster degenerates to a single tag
group, which is the form we use here).
"""
import pathlib 
from typing import Dict, List, Optional, Tuple
import numpy as np

from saga import Schedule, Scheduler, ScheduledTask, TaskGraph, Network

thisdir = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Cost model (eqs. 1 and 2 from the paper)
# ---------------------------------------------------------------------------

def _compute_time(task_graph: TaskGraph, network: Network,
                  task_name: str, node_name: str) -> float:
    """Average time to compute a task on a node (eq. 1)."""
    return task_graph.get_task(task_name).cost / network.get_node(node_name).speed


def _transfer_time(task_graph: TaskGraph, network: Network,
                   src_task: str, dst_task: str,
                   src_node: str, dst_node: str) -> float:
    """Average time to transfer a tuple between two nodes (eq. 2).

    Returns 0 if both tasks are placed on the same node (no inter-node transfer).
    """
    if src_node == dst_node:
        return 0.0
    # Edge weight in the task graph carries the data size; bandwidth on
    # network edge carries link capacity. Self-loop on a network node has
    # infinite/very-large bandwidth in saga, so this still works if you ever
    # call it with src==dst.
    data_size = task_graph.get_dependency(src_task, dst_task).size
    bandwidth = network.get_edge(src_node, dst_node).speed
    return data_size / bandwidth


# ---------------------------------------------------------------------------
# Critical path identification (longest path under homogeneous assumption)
# ---------------------------------------------------------------------------

def _average_compute(task_graph: TaskGraph, network: Network, task_name: str) -> float:
    """Average compute time of a task across all nodes (homogeneous proxy)."""
    speeds = [n.speed for n in network.nodes]
    avg_speed = sum(speeds) / len(speeds)
    return task_graph.get_task(task_name).cost / avg_speed


def _average_transfer(task_graph: TaskGraph, network: Network,
                      src_task: str, dst_task: str) -> float:
    """Average transfer time across all distinct-node pairs (homogeneous proxy)."""
    bws = []
    nodes = list(network.nodes)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            bws.append(network.get_edge(u.name, v.name).speed)
    if not bws:
        return 0.0
    avg_bw = sum(bws) / len(bws)
    data_size = task_graph.get_dependency(src_task, dst_task).size
    return data_size / avg_bw


def _critical_path(task_graph: TaskGraph, network: Network) -> List[str]:
    """Return the longest-weight path through the DAG.

    Weights use averaged compute + transfer costs (homogeneous-network
    simplification, per the paper's note on p. 9616).
    """
    topo = [n.name for n in task_graph.topological_sort()]

    # longest[t] = (length of longest path ending at t, predecessor on that path)
    longest: Dict[str, Tuple[float, Optional[str]]] = {}
    for t in topo:
        node_cost = _average_compute(task_graph, network, t)
        best_len = node_cost
        best_pred: Optional[str] = None
        for edge in task_graph.in_edges(t):
            pred = edge.source
            edge_cost = _average_transfer(task_graph, network, pred, t)
            cand = longest[pred][0] + edge_cost + node_cost
            if cand > best_len:
                best_len = cand
                best_pred = pred
        longest[t] = (best_len, best_pred)

    # Find sink with maximum length; trace back.
    sink = max(longest, key=lambda t: longest[t][0])
    path: List[str] = []
    cur: Optional[str] = sink
    while cur is not None:
        path.append(cur)
        cur = longest[cur][1]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Critical-path mapping via dynamic programming (eq. 4 / Algorithm 2)
# ---------------------------------------------------------------------------

def _map_critical_path(critical_path: List[str],
                       network: Network,
                       task_graph: TaskGraph) -> Dict[str, str]:
    """Map critical-path tasks to nodes minimizing the max stage time.

    Implements the 2D DP from eq. 4. State Pmap[j][i] = minimum achievable
    bottleneck-stage time for placing the first j+1 CP tasks on a path that
    ends with CP-task j on node i. Transitions consider:
      Case I:  task j placed on the same node as task j-1 (no transfer)
      Case II: task j placed on any neighbor of the node holding task j-1
               (incurs transfer cost)
    """
    nodes = [n.name for n in network.nodes]
    n_tasks = len(critical_path)

    # Pmap[j][node] = (best bottleneck so far, predecessor node on CP)
    pmap: List[Dict[str, Tuple[float, Optional[str]]]] = [
        {nd: (np.inf, None) for nd in nodes} for _ in range(n_tasks)
    ]

    # Base case: place first CP task on each candidate node.
    first_task = critical_path[0]
    for nd in nodes:
        pmap[0][nd] = (_compute_time(task_graph, network, first_task, nd), None)

    # Fill DP.
    for j in range(1, n_tasks):
        cur_task = critical_path[j]
        prev_task = critical_path[j - 1]
        for nd in nodes:
            cur_compute = _compute_time(task_graph, network, cur_task, nd)
            best_bottleneck = np.inf
            best_prev_node: Optional[str] = None

            for prev_nd in nodes:
                prev_bottleneck = pmap[j - 1][prev_nd][0]
                if prev_bottleneck == np.inf:
                    continue

                if prev_nd == nd:
                    # Case I: same node, no transfer.
                    stage_time = cur_compute
                else:
                    # Case II: different node, pay transfer.
                    transfer = _transfer_time(
                        task_graph, network, prev_task, cur_task, prev_nd, nd
                    )
                    stage_time = max(cur_compute, transfer)

                # Bottleneck is max over all stages so far.
                cand = max(prev_bottleneck, stage_time)
                if cand < best_bottleneck:
                    best_bottleneck = cand
                    best_prev_node = prev_nd

            pmap[j][nd] = (best_bottleneck, best_prev_node)

    # Trace back the best assignment.
    final_node = min(nodes, key=lambda nd: pmap[n_tasks - 1][nd][0])
    assignment: Dict[str, str] = {}
    cur_node: Optional[str] = final_node
    for j in range(n_tasks - 1, -1, -1):
        assert cur_node is not None
        assignment[critical_path[j]] = cur_node
        cur_node = pmap[j][cur_node][1]

    return assignment


# ---------------------------------------------------------------------------
# Non-CP greedy layer-oriented mapping
# ---------------------------------------------------------------------------

def _map_non_critical(non_cp_tasks: List[str],
                      cp_assignment: Dict[str, str],
                      network: Network,
                      task_graph: TaskGraph) -> Dict[str, str]:
    """Place non-CP tasks layer by layer, heaviest first, on best node.

    "Best" = node minimizing (compute_time + total transfer from already-placed
    predecessors). Heaviest = highest avg compute + avg transfer cost.
    """
    assignment = dict(cp_assignment)

    # Sort non-CP tasks within their topological layer by descending cost.
    def task_weight(t: str) -> float:
        c = _average_compute(task_graph, network, t)
        x = sum(_average_transfer(task_graph, network, e.source, t)
                for e in task_graph.in_edges(t))
        return c + x

    # Process in topological order so predecessors are already placed when
    # we make our decision; within that constraint, prefer heavier tasks.
    topo = [n.name for n in task_graph.topological_sort()]
    pending = [t for t in topo if t in non_cp_tasks]
    pending.sort(key=task_weight, reverse=True)
    # Re-sort so topological order is still respected (stable sort by topo idx).
    topo_idx = {t: i for i, t in enumerate(topo)}
    pending.sort(key=lambda t: topo_idx[t])

    for t in pending:
        best_node = None
        best_cost = np.inf
        for nd in network.nodes:
            comp = _compute_time(task_graph, network, t, nd.name)
            transfer = 0.0
            for edge in task_graph.in_edges(t):
                pred = edge.source
                if pred in assignment:
                    transfer += _transfer_time(
                        task_graph, network, pred, t,
                        assignment[pred], nd.name
                    )
            cost = comp + transfer
            if cost < best_cost:
                best_cost = cost
                best_node = nd.name
        assignment[t] = best_node

    return assignment


# ---------------------------------------------------------------------------
# Scheduler class
# ---------------------------------------------------------------------------

class MTScheduler(Scheduler):
    """Maximum-Throughput Scheduler for heterogeneous DAG scheduling.

    Source: Al-Sinayyid & Zhu (2020), https://doi.org/10.1007/s11227-020-03223-z

    Adapted from streaming/Apache Storm context to static DAG scheduling. The
    algorithm minimizes the bottleneck stage time (max compute + transfer over
    all stages of the critical path), which equals 1/throughput for a DAG
    executed repeatedly in pipelined fashion.
    """

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Schedule tasks on the network to maximize pipelined throughput.

        Args:
            network: The network graph (heterogeneous nodes + bandwidth links).
            task_graph: The task graph (DAG of computations + data transfers).
            schedule: Existing schedule to extend (optional).
            min_start_time: Minimum start time for any task.

        Returns:
            A Schedule mapping each task to a node with start/end times.
        """
        # Phase 1: identify the critical path (longest weighted path in DAG).
        critical_path = _critical_path(task_graph, network)

        # Phase 2: DP-based heterogeneity-aware mapping for CP tasks.
        cp_assignment = _map_critical_path(critical_path, network, task_graph)

        # Phase 3: greedy layer-oriented placement for non-CP tasks.
        all_tasks = [n.name for n in task_graph.topological_sort()]
        non_cp_tasks = [t for t in all_tasks if t not in set(critical_path)]
        full_assignment = _map_non_critical(
            non_cp_tasks, cp_assignment, network, task_graph
        )

        # Phase 4: realize the assignment as a Schedule with concrete times,
        # using the same earliest-start-time logic HEFT uses.
        schedule = Schedule(task_graph, network)
        schedule_order = [n.name for n in task_graph.topological_sort()]

        for task_name in schedule_order:
            if schedule.is_scheduled(task_name):
                continue
            assigned_node = full_assignment[task_name]
            start_time = schedule.get_earliest_start_time(
                task=task_name, node=assigned_node, append_only=False
            )
            start_time = max(start_time, min_start_time)
            runtime = (
                task_graph.get_task(task_name).cost
                / network.get_node(assigned_node).speed
            )
            finish_time = start_time + runtime
            new_task = ScheduledTask(
                node=assigned_node,
                name=task_name,
                start=start_time,
                end=finish_time,
            )
            schedule.add_task(new_task)

        return schedule