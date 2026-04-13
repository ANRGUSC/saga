# SAGA

[![CI](https://github.com/ANRGUSC/saga/actions/workflows/ci.yml/badge.svg)](https://github.com/ANRGUSC/saga/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/anrg-saga.svg)](https://badge.fury.io/py/anrg-saga)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

SAGA: **S**cheduling **A**lgorithms **Ga**thered.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/ANRGUSC/saga?quickstart=1)

## Introduction

SAGA – Scheduling Algorithms Gathered – is a Python toolkit/library for designing, comparing, and visualising DAG-based computational workflow-scheduler performance on heterogeneous compute networks (also known as dispersed computing).
It ships with a collection of scheduling algorithms, including classic heuristics (HEFT, CPOP), brute-force baselines, SMT-based optimisers, and more, all under one cohesive API.

The algorithms are all implemented in Python using a common interface.  Scripts for validating and comparing the performance of the algorithms are also provided.


## Prerequisites

### Python Version

All components of this repository have been tested with **Python 3.11**. To ensure compatibility and ease of environment management, we recommend using **[Conda](https://docs.conda.io/en/latest/)**.

To create a new Conda environment with Python 3.11:

```bash
conda create -n saga-env python=3.11
conda activate saga-env
```

For more information on managing Python versions with Conda, refer to the [Conda documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-python.html). ([Managing Python — conda 25.3.0 documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-python.html?utm_source=chatgpt.com))

## Usage

### Installation

#### Local Installation

Clone the repository and install the requirements:

```bash
git clone https://github.com/ANRGUSC/saga.git
cd saga
pip install -e .
```

### Running the Tests

Unit tests generate random task graphs and networks to verify scheduler correctness. They also check the RandomVariable utilities used for stochastic scheduling.

#### Locally

You can run the tests using `pytest`:

```bash
pytest ./tests
```

You may want to skip some of the tests that are too slow.
You can do this ddirectly:
```bash
pytest ./tests -k "not (branching and (BruteForceScheduler or SMTScheduler))"
```

or by setting a timeout for the tests:

```bash
pytest ./tests --timeout=60
```

To run a specific test or scheduler-task combination, use the `-k` option. For example, to run the `HeftScheduler` tests on the `diamond` task graph:

```bash
pytest ./tests -k "HeftScheduler and diamond"
```

### Linting and Type Checking

The CI pipeline also runs a linter and type checker. You can run these locally:

```bash
# Lint with ruff
ruff check src/saga

# Check formatting with ruff
ruff format --check src/saga

# Type check with mypy
mypy src/saga --ignore-missing-imports
```

To auto-fix lint issues or reformat code:

```bash
ruff check src/saga --fix
ruff format src/saga
```

### Running the Algorithms

The algorithms are implemented as Python modules. The following example shows how to run the HEFT algorithm on a workflow:

```python
from saga.schedulers import HeftScheduler

scheduler = HeftScheduler()
network: Network = ...
task_graph: TaskGraph = ...
scheduler.schedule(network, task_graph)
```

### Examples

The repository contains several example scripts illustrating different algorithms and scenarios.
You can find them under [scripts/examples](./scripts/examples). To run an example, use:

```bash
python scripts/examples/<example_name>/main.py
```

The table of contents in `scripts/examples/Readme.md` lists examples ranging from basic usage to dynamic networks and scheduler comparisons.


### Real-World Execution (Makeflow / Work Queue / Chameleon)

In addition to simulating scheduling algorithms, SAGA can now execute actual Python task graphs on real compute clusters.  Users define tasks with a Dask/Parsl-style decorator, SAGA picks the placement, and the emitted workflow runs on CCTools Makeflow or Work Queue — with per-node pinning honoured by the cluster.

```python
from saga.execution import saga_task, SagaGraph
from saga.execution.chameleon import ChameleonNode, build_network
from saga.execution.makeflow import emit_makeflow
from saga.execution.workqueue import WorkQueueRunner
from saga.schedulers import HeftScheduler

@saga_task(inputs=[], outputs=["x"], cost=lambda cfg: cfg["N"])
def generate(cfg): ...

@saga_task(inputs=["x"], outputs=["y"], cost=lambda cfg: 10 * cfg["N"])
def transform(x, cfg): ...

graph = SagaGraph([generate, transform], config={"N": 10_000})
network = build_network([ChameleonNode("worker-0", 2.0), ChameleonNode("worker-1", 1.0)])
schedule = HeftScheduler().schedule(network, graph.to_task_graph())

# Emit a Makeflow JSON for portable / local execution, or run on Work Queue
# with hard per-node pinning via specify_feature.
emit_makeflow(graph, schedule, "out/", task_module="my_pkg.tasks")
WorkQueueRunner(graph, schedule, task_module="my_pkg.tasks").run()
```

See [scripts/examples/dagprofiler_chameleon](./scripts/examples/dagprofiler_chameleon) for a full end-to-end example and `provision_workers.sh` helper for launching CCTools workers on a Chameleon Cloud lease.  SAGA also integrates with [dagprofiler](https://github.com/ANRGUSC/dagprofiler) via `saga.execution.profile_to_task_graph(...)` so you can drive the scheduler from measured runtimes and edge data sizes.

### Experiments
To reproduce the experiments from papers using SAGA, see the [experiments](./scripts/experiments) directory.

### Reference
A research paper that goes with this repo and that contains useful details is available online at [ArXiV](https://arxiv.org/pdf/2403.07120).


### Acknowledgements

This work was supported in part by Army Research Laboratory under Cooperative Agreement [W911NF-17-2-0196](https://www.usaspending.gov/award/ASST_NON_W911NF1720196_097).

This material is based upon work supported by the National Science Foundation under Award No. [2451267](https://www.nsf.gov/awardsearch/show-award?AWD_ID=2451267).
