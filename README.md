# Saga

Saga: **S**cheduling **A**lgorithms **Ga**thered.

## Introduction

This repository contains a collection of scheduling algorithms.  
The algorithms are implemented in Python using a common interface.  
Scripts for validating and comparing the performance of the algorithms are also provided.

## Usage

### Installation

#### Local Installation

Clone the repository and install the requirements:

```bash
git clone https://github.com/ANRGUSC/saga.git
cd saga
pip install -e ./src
```

To install additional dependencies like `pytest` for running tests, run:

```bash
pip install pytest pytest-timeout
```

Some of the algorithms might rely on external solvers, such as Z3. To install Z3 and configure `pysmt`, use:

```bash
pip install pysmt
pysmt-install --z3
```

#### Docker Installation

You can also run Saga using Docker. The provided `Dockerfile` will handle all dependencies, including solvers and testing tools.

1. Build the Docker image:
   ```bash
   docker build -t saga-schedulers .
   ```

2. Run the image:
   ```bash
   docker run --rm saga-schedulers
   ```

By default, the Docker image will run the tests when started.

### Running the Tests

#### Locally

You can run the tests using `pytest`. Make sure you have installed the necessary dependencies, including `pytest` and `pytest-timeout`:

```bash
pip install pytest pytest-timeout
```

Then, run the tests:

```bash
pytest
```

If you want to specify a timeout for the tests, you can do so using the `--timeout` option. For example:

```bash
pytest --timeout=60
```

To run a specific test or scheduler-task combination, use the `-k` option. For example, to run the `HeftScheduler` tests on the `diamond` task graph:

```bash
pytest -k "HeftScheduler and diamond"
```

#### Using Docker

When running the Docker image, the tests will run automatically. You can also pass specific `pytest` options when running the Docker container.

For example, to run all tests with a 120-second timeout:

```bash
docker run --rm saga-schedulers pytest --timeout=120
```

Or to run a specific test combination (e.g., `HeftScheduler` and `diamond`):

```bash
docker run --rm saga-schedulers pytest -k "HeftScheduler and diamond"
```

### Running the Algorithms

The algorithms are implemented as Python modules. The following example shows how to run the HEFT algorithm on a workflow:

```python
from saga.schedulers import HeftScheduler

scheduler = HeftScheduler()
network: nx.Graph = ...
task_graph: nx.DiGraph = ...
scheduler.schedule(network, task_graph)
```

See the examples in the [examples](./scripts/examples) directory for more!

### Experiments from Papers

To reproduce the experiments from papers using SAGA, see the [experiments](./scripts/experiments) directory.
