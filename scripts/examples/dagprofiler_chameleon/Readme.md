# Real-world execution: SAGA → Makeflow / Work Queue on Chameleon Cloud

This example shows the full pipeline from "Python task graph" to "actual
execution on heterogeneous compute nodes":

1. **Define** a task graph with the `@saga_task` decorator (see `main.py`).
2. **Describe** the target cluster with `ChameleonNode` + `build_network`.
3. **Schedule** with any SAGA scheduler (e.g. HEFT).
4. **Execute** via either
    - Makeflow JSON (`makeflow --jx …`), or
    - CCTools Work Queue with per-node pinning (`WorkQueueRunner`).

## Files

| File | Purpose |
|---|---|
| `main.py` | End-to-end driver: defines tasks, schedules, emits / runs. |
| `provision_workers.sh` | SSH-out helper that starts one `work_queue_worker` per reserved Chameleon node with a unique `--feature` tag. |
| `Readme.md` | This file. |

## Running locally (no cluster required)

```bash
python scripts/examples/dagprofiler_chameleon/main.py --mode emit
```

This writes a Makeflow JSON and a staging dir.  You can then run the workflow
locally with the `makeflow` binary:

```bash
makeflow --jx ./saga_chameleon_out/workflow.json -T local
```

## Running on Chameleon via Work Queue

Prerequisites:

- A Chameleon lease with N bare-metal nodes reachable via SSH.
- CCTools installed on each node (the easiest route is
  `conda install -c conda-forge ndcctools`).
- `work_queue` Python bindings on the driver machine (same package).

```bash
# 1. On the driver machine, launch the SAGA Work Queue manager + submit tasks.
python scripts/examples/dagprofiler_chameleon/main.py \
    --mode wq --project saga-demo --port 9123

# 2. From another shell (or cron / a tmux pane), start one worker per node.
#    The feature string must match SAGA's convention
#    ``saga-node-<node_name_with_nonalnum_as_underscores>``.
bash scripts/examples/dagprofiler_chameleon/provision_workers.sh \
    saga-demo saga-worker-0 saga-worker-1 saga-worker-2
```

## How pinning works

Work Queue has two concepts we rely on:

1. A worker advertises `--feature <name>` on start-up.
2. A task calls `task.specify_feature(<name>)` — it will **only** run on a
   worker with that feature.

`saga.execution.workqueue.WorkQueueRunner` uses `feature_for(node_name)` on
both sides, so every SAGA schedule maps 1-to-1 onto pinned Work Queue tasks.

## Why Makeflow JSON *and* Work Queue?

Makeflow's JSON schema exposes per-rule `category` but not per-task feature
affinity — it's primarily a way to describe resource requirements, not
worker affinity.  We therefore ship **two** back-ends:

| Back-end | Pinning? | Use when |
|---|---|---|
| `emit_makeflow(...)` | soft (category-per-node; Makeflow assigns by resources) | You want a portable, inspectable workflow, or you'll run locally / on a cluster where Makeflow's default scheduling is acceptable. |
| `WorkQueueRunner(...).run()` | hard (`specify_feature` per task) | You want SAGA to make the placement decisions and the platform to honour them exactly. |

## Using `dagprofiler`

If you've profiled your real Python code with
[dagprofiler](https://github.com/ANRGUSC/dagprofiler) (producing a
`profile.json`), you can feed the measured costs straight into SAGA:

```python
from saga.execution import profile_to_task_graph
task_graph = profile_to_task_graph(
    "profile.json",
    use_measured_time=True,   # use measured compute_time_ms as the cost
    edge_units="bytes",
)
```

Pair it with the same `Network` (from `build_network`) and any SAGA
scheduler, and drive execution with `WorkQueueRunner` as above.
