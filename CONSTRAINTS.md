# Task Placement Constraints in SAGA

This document describes the `Constraints` API introduced to generalise the
task-placement ("pinning") capability in SAGA.

---

## Background

### Earlier research version

An earlier research script used a complementary *exclusion* model, storing
forbidden nodes as a graph-level attribute:

```python
task_graph.graph["schedule_restrictions"] = {
    "t_1": {"v_2", "v_3"},  # t_1 must NOT run on v_2 or v_3
}
```

Constraints were enforced by injecting ±1e9 magic penalties into comparison
functions passed to `ConstrainedGreedyInsert`. While flexible, this approach
had a silent failure mode: if *every* node was penalised for a task, the
scheduler still picked one and silently produced an infeasible schedule, which
had to be caught by a post-hoc validation call.

---

## The `Constraints` API

### Module

```
src/saga/constraints.py
```

### Core classes

#### `Constraint` (abstract base)

```python
class Constraint(BaseModel, ABC):
    task: str   # name of the task this constraint applies to

    @abstractmethod
    def allowed_nodes(self, node_names: List[str]) -> List[str]: ...
```

#### `AllowedNodes` — allow-list constraint

The task may **only** run on the specified nodes.

```python
AllowedNodes(task="t_1", nodes=["v_1"])           # exact pin
AllowedNodes(task="t_2", nodes=["v_1", "v_2"])    # restrict to a subset
```

#### `ForbiddenNodes` — exclude-list constraint

The task must **not** run on the specified nodes.
This generalises the old `schedule_restrictions` model with a proper
first-class object instead of a graph attribute.

```python
ForbiddenNodes(task="t_3", nodes=["v_3"])         # exclude v_3
ForbiddenNodes(task="t_4", nodes=["v_1", "v_2"])  # exclude v_1 and v_2
```

#### `Constraints` — composite container

Holds a list of `Constraint` objects and exposes a single resolution method.

```python
class Constraints(BaseModel):
    constraints: List[Constraint]

    def get_candidate_nodes(self, task: str, node_names: List[str]) -> List[str]:
        ...

    @classmethod
    def from_task_graph(cls, task_graph) -> "Constraints":
        ...
```

`get_candidate_nodes` raises `ValueError` if a constraint yields an empty
candidate set, making infeasibility explicit and immediate rather than a
silent bad schedule.

---

## Using Constraints

### With `ParametricScheduler`

Pass a `Constraints` instance as the `constraints` keyword argument.

```python
from saga.constraints import AllowedNodes, ForbiddenNodes, Constraints
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import GreedyInsert, GreedyInsertCompareFuncs, UpwardRanking

constraints = Constraints(constraints=[
    AllowedNodes(task="t_1",  nodes=["v_1"]),           # pin t_1 to v_1
    AllowedNodes(task="t_2",  nodes=["v_1", "v_2"]),    # restrict t_2 to v_1 or v_2
    ForbiddenNodes(task="t_3", nodes=["v_3"]),           # t_3 cannot run on v_3
])

scheduler = ParametricScheduler(
    initial_priority=UpwardRanking(),
    insert_task=GreedyInsert(
        append_only=False,
        compare=GreedyInsertCompareFuncs.EFT,
    ),
)
schedule = scheduler.schedule(network, task_graph, constraints=constraints)
```

The `constraints` parameter is forwarded into every `InsertTask.call()` via
the existing `nodes` parameter — no changes to `GreedyInsert` were required.

The same applies to `ParametricSufferageScheduler`, which additionally
ensures that the "second best" sufferage candidate set is also constrained
(and handles the degenerate case where only one node is allowed gracefully).

---

## File Changes Summary

| File | Change |
|---|---|
| `src/saga/constraints.py` | **New** — `Constraint`, `AllowedNodes`, `ForbiddenNodes`, `Constraints` |
| `src/saga/__init__.py` | **Bug fix** — `get_earliest_start_time`: `math.inf` edge speed (same-node self-loop) was incorrectly treated as unreachable; now correctly yields zero communication cost |
| `src/saga/schedulers/heft.py` | Added `Constraints.from_task_graph()` |
| `src/saga/schedulers/mct.py` | Added `Constraints.from_task_graph()` |
| `src/saga/schedulers/minmin.py` | Replaced inner `get_candidate_nodes` with `Constraints.from_task_graph()` |
| `src/saga/schedulers/maxmin.py` | Replaced inner `get_candidate_nodes` with `Constraints.from_task_graph()` |
| `src/saga/schedulers/met.py` | Added `Constraints.from_task_graph()` to filter candidate nodes per task |
| `src/saga/schedulers/olb.py` | Added `Constraints.from_task_graph()` to restrict next-available-node selection per task |
| `src/saga/schedulers/sufferage.py` | Added per-task candidate filtering for both ECT computation and node selection |
| `src/saga/schedulers/wba.py` | Added per-task candidate filtering in the makespan-increase inner loop |
| `src/saga/schedulers/dps.py` | Added per-task candidate filtering in the EFT node selection loop |
| `src/saga/schedulers/mst.py` | Added per-task candidate filtering; cluster decisions still respected when within allowed set |
| `src/saga/schedulers/msbc.py` | Added per-task candidate filtering over speed-sorted node list |
| `src/saga/schedulers/gdl.py` | Applied constraints to `preferred_node` and `cost` cached closures |
| `src/saga/schedulers/bil.py` | Applied constraints when building BIM lists and revised BIMs for node selection |
| `src/saga/schedulers/cpop.py` | Applied constraints for both critical-path and non-critical-path tasks |
| `src/saga/schedulers/fcp.py` | Applied constraints to both `p_start` and `p_arrive` in `select_processor` |
| `src/saga/schedulers/etf.py` | Added per-task candidate filtering inside `_get_start_times` |
| `src/saga/schedulers/brute_force.py` | Generates mappings only over per-task candidate nodes (reduces search space) |
| `src/saga/schedulers/fastest_node.py` | Selects fastest node among each task's allowed candidates (may differ per task) |
| `src/saga/schedulers/hbmct.py` | Passed constraints to `get_initial_assignments` and the group re-balancing loop |
| `src/saga/schedulers/smt.py` | Restricted `ExactlyOne` per task to allowed nodes; forbidden nodes forced to negative start time |
| `src/saga/schedulers/flb.py` | Import added; full per-task filtering not implemented — FLB's enabling-processor queue structure is architecturally incompatible with arbitrary per-task node constraints |
| `src/saga/schedulers/duplex.py` | No changes needed — delegates to MinMin/MaxMin which already support constraints |
| `src/saga/schedulers/hybrid.py` | No changes needed — delegates to sub-schedulers which already support constraints |
| `src/saga/schedulers/parametric/__init__.py` | Added `constraints` param to `ParametricScheduler.schedule()` |
| `src/saga/schedulers/parametric/components.py` | Added `constraints` param to `ParametricSufferageScheduler.schedule()`; sufferage second-best candidate set is now also constrained |
| `constraints_example.py` | **New** — runnable demo of all constraint scenarios |

### Bug fix: `get_earliest_start_time` and infinite-speed self-loops

`Network.create()` assigns `math.inf` speed to self-loop edges (same-node
communication, i.e. zero transfer cost). The original `get_earliest_start_time`
had:

```python
if network_edge.speed <= 0 or not math.isfinite(network_edge.speed):
    arrival_time = math.inf
```

The `not math.isfinite(inf)` branch incorrectly made same-node communication
unreachable, producing `inf` makespans whenever a task was placed on the same
node as one of its parents. The corrected logic:

```python
if network_edge.speed <= 0:
    arrival_time = math.inf        # no link — unreachable
elif math.isinf(network_edge.speed):
    arrival_time = parent_task.end # same node — zero transfer cost
else:
    arrival_time = parent_task.end + (dependency.size / network_edge.speed)
```

This bug was latent in the original code because the existing test suite used
`weight=1e9` (large finite) for self-loops rather than the `math.inf` default
from `Network.create()`. The Constraints API surfaced it by forcing co-location
of parents and children onto restricted nodes.

---

## Example Script

```
python constraints_example.py           # print schedule summaries
python constraints_example.py --gantt   # also save Gantt chart PNGs
```

The script covers four scenarios:

1. **Unconstrained** — baseline, no restrictions.
2. **AllowedNodes** — pin `t_1` to `v_1`, restrict `t_4` to `{v_2, v_3}`.
3. **ForbiddenNodes** — forbid `t_2` from `v_1`, forbid `t_3` from `v_3`.
4. **Mixed** — combine both constraint types across all available schedulers.

---

## Extending the API

To add a new constraint type, subclass `Constraint` and implement
`allowed_nodes`:

```python
from saga.constraints import Constraint

class PreferFastestNode(Constraint):
    """Allow only the single fastest node in the network."""

    def allowed_nodes(self, node_names: List[str]) -> List[str]:
        # network not available here; resolve at call site or override
        # get_candidate_nodes in a subclass of Constraints instead.
        return node_names  # no-op fallback
```

For constraints that require access to the `Network` object (e.g. node
speeds, topology), override `Constraints.get_candidate_nodes` in a subclass
to pass the network through.