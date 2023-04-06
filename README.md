# Saga
Saga: **S**cheduling **A**lgorithms **Ga**thered.

## Introduction
This repository contains a collection of scheduling algorithms. 
The algorithms are implemented in python using a common interface.
Scripts for validating the schedules produced by the algorithms are also provided.
Scripts for comparing the performance of the algorithms are also provided.

## Algorithms
The following algorithms are implemented:
* Common:
    * HEFT: Heteregeneous Earliest Finish Time
    * CPoP: Critical Path on Processor
    * FastestNode: Schedule all tasks on the fastest node
* Stochastic: (stochastic task cost, data size, compute speed, and communication strength)
    * SHEFT
    * Improved SHEFT
    * Stochastic HEFT
    * Mean HEFT

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
from saga.common.heft import HeftScheduler

scheduler = HeftScheduler()
network: nx.Graph = ...
task_graph: nx.DiGraph = ...
scheduler.schedule(network, task_graph)
```
