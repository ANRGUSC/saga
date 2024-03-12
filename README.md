# Saga

Saga: **S**cheduling **A**lgorithms **Ga**thered.

## Introduction

This repository contains a collection of scheduling algorithms.
The algorithms are implemented in python using a common interface.
Scripts for validating the schedules produced by the algorithms are also provided.
Scripts for comparing the performance of the algorithms are also provided.

## Usage

### Installation

Clone the repository and install the requirements:

```bash
pip install anrg.saga
```

### Running the algorithms

The algorithms are implemented as python modules.
The following example shows how to run the HEFT algorithm on a workflow:

```python
from saga.schedulers import HeftScheduler

scheduler = HeftScheduler()
network: nx.Graph = ...
task_graph: nx.DiGraph = ...
scheduler.schedule(network, task_graph)
```

See [./examples/heft/main.py](./examples/heft/main.py) for a complete example.
