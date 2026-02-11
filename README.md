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


### Experiments
To reproduce the experiments from papers using SAGA, see the [experiments](./scripts/experiments) directory.


### Acknowledgements

This work was supported in part by Army Research Laboratory under Cooperative Agreement [W911NF-17-2-0196](https://www.usaspending.gov/award/ASST_NON_W911NF1720196_097).
This material is based upon work supported by the National Science Foundation under Award No. [2451267](https://www.nsf.gov/awardsearch/show-award?AWD_ID=2451267).