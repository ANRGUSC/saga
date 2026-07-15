# SAGA API quick reference (for building families)

Everything lives in the `saga` package (installed in `saga/.venv`). Run scripts with
`saga/.venv/bin/python`. Import the library from anywhere once that interpreter is used.

## Core objects

```python
from saga import Network, TaskGraph, Scheduler, Schedule

# Network is UNDIRECTED and fully connected. Node speed = compute speed;
# edge speed = bandwidth. Higher speed = faster.
network = Network.create(
    nodes=[("0", 0.8), ("1", 0.3)],          # (name, speed)
    edges=[("0", "1", 5.0)],                  # (src, tgt, speed); add each pair ONCE.
)                                             # missing pairs default to speed 0;
                                              # self-loops default to inf (intra-node comm is free).

# TaskGraph is a DAG. Task cost = compute work; dependency size = data to transfer.
task_graph = TaskGraph.create(
    tasks=[("A", 1.0), ("B", 0.5)],           # (name, cost)
    dependencies=[("A", "B", 0.2)],           # (src, tgt, size); (src, tgt) => size 0
)
# If the DAG has multiple sources/sinks, TaskGraph.create auto-adds zero-cost
# __super_source__ / __super_sink__ nodes (logs a warning; harmless).
```

Time to run task *t* on node *n* = `t.cost / n.speed`.
Time to send dependency data of size *s* over link (u,v) = `s / edge.speed`. Same-node = 0.

## Scheduling and makespan

```python
from saga.pisa import SCHEDULERS            # dict: name -> Scheduler instance
sched = SCHEDULERS["HEFT"]                  # or resolve_scheduler() in family_lib
schedule = sched.schedule(network, task_graph)
ms = schedule.makespan                       # float; lower is better
```

Valid scheduler names (the PISA registry):
`BIL, CPoP, Duplex, ETF, FCP, FLB, FastestNode, GDL, HEFT, MCT, MET, MaxMin, MinMin, OLB, WBA, Sufferage`.

## Homogeneity constraints (IMPORTANT)

Some schedulers are only defined on restricted instances. If either algorithm in
your pair is listed, your family MUST respect the constraint or results are invalid:

- Homogeneous COMPUTE required (all node speeds equal): `ETF, FCP, FLB`
- Homogeneous COMMUNICATION required (all edge speeds equal): `BIL, GDL, FCP, FLB`

`seed_pisa.py` already enforces these; mirror them in your generator.

## Useful generators and knobs

```python
from saga.utils.random_graphs import (
    get_network, get_chain_dag, get_diamond_dag, get_fork_dag, get_branching_dag,
)
from saga.utils.random_variable import UniformRandomVariable, RandomVariable

# get_network(num_nodes, node_weight_distribution=..., edge_weight_distribution=...)
# get_branching_dag(levels, branching_factor, node_weight_distribution=..., edge_weight_distribution=...)
# RandomVariable(samples=[1.0]) => constant (use for homogeneous dimensions).

# Control communication-to-computation ratio (returns a NEW scaled network):
network = network.scale_to_ccr(task_graph, target_ccr=5.0)   # high CCR => comm-dominated
```

CCR is the single most useful global knob: many algorithm gaps only appear at very
low CCR (compute-dominated) or very high CCR (comm-dominated). Sweep it.

`seed_pisa.py`'s search only reweights/rewires a fixed-size graph — it cannot add or
remove tasks or nodes mid-search (the underlying PISA `Change` types in
`saga.pisa.changes` are limited to add/delete dependency and reweight task/dependency/
node/edge). The task/node count is fixed by `--init`/`--nodes` for the entire run, so a
near-1.0 best ratio can mean the graph is simply too small/large, not that no gap exists.

## What each algorithm roughly does (intuition for hypotheses)

- **HEFT / CPoP**: list schedulers, rank tasks by upward path length, place each on the
  node giving earliest finish. Good at exploiting parallelism and heterogeneity.
- **FastestNode**: puts every task on the single fastest node (ignores parallelism).
- **MET**: minimum execution time — picks fastest node per task ignoring contention.
- **MCT / OLB**: minimum completion time / opportunistic load balancing (greedy, myopic).
- **MinMin / MaxMin / Sufferage**: batch-mode heuristics over ready tasks.
- **Duplex** = min(MinMin, MaxMin). **ETF/FCP/FLB/BIL/GDL**: classic list heuristics
  (homogeneous variants; see constraints above).

Exploit the loser's blind spot: e.g. FastestNode ignores parallelism, so wide DAGs
with cheap communication crush it; MET ignores contention, so many tasks preferring
one node crush it.
