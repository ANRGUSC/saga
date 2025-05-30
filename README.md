# Saga

Saga: **S**cheduling **A**lgorithms **Ga**thered.

## Introduction

Saga – Scheduling Algorithms Gathered – is a Python toolkit/library for designing, comparing, and visualising DAG-based computational workflow-scheduler performance on heterogeneous compute networks (also known as dispersed computing).
It ships with a collection of scheduling algorithms, including classic heuristics (HEFT, CPOP), brute-force baselines, SMT-based optimisers, and more, all under one cohesive API.

The algorithms are all implemented in Python using a common interface.  Scripts for validating and comparing the performance of the algorithms are also provided.

## Package Layout

The source code resides in `src/saga` and is organized as follows:

- **`scheduler.py`** – defines the abstract `Scheduler` base class and the `Task` dataclass.
- **`schedulers/`** – implementations of various algorithms such as HEFT, CPoP and others,
  all exposed through `saga.schedulers`.
- **`utils/`** – helper modules for generating task graphs, drawing Gantt charts and
  validating schedules. It also includes `simulator.py` for stochastic simulations
  and `data.py` for datasets used in experiments.


## Prerequisites

### Python Version

All components of this repository have been tested with **Python 3.11**. To ensure compatibility and ease of environment management, we recommend using **[Conda](https://docs.conda.io/en/latest/)**.

To create a new Conda environment with Python 3.11:


```bash
conda create -n saga-env python=3.11
conda activate saga-env
```


For more information on managing Python versions with Conda, refer to the [Conda documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-python.html). ([Managing Python — conda 25.3.0 documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-python.html?utm_source=chatgpt.com))

### Graphviz Installation

To enable task graph visualization, ensure that **Graphviz** is installed on your system. Graphviz provides the `dot` command-line tool used for rendering graphs.

#### Installation via Conda (Recommended)

You can install Graphviz and its Python interface using Conda: ([anaconda - graphviz - can't import after installation - Stack Overflow](https://stackoverflow.com/questions/33433274/anaconda-graphviz-cant-import-after-installation?utm_source=chatgpt.com))


```bash
conda install -c conda-forge graphviz python-graphviz
```


This command installs both the Graphviz binaries and the `python-graphviz` package, facilitating seamless integration with Python scripts. ([anaconda - graphviz - can't import after installation - Stack Overflow](https://stackoverflow.com/questions/33433274/anaconda-graphviz-cant-import-after-installation?utm_source=chatgpt.com))

#### Manual Installation

If you prefer manual installation:

- **macOS**:

  - Using [Homebrew](https://brew.sh/):

    ```bash
    brew install graphviz
    ```

  - Using [MacPorts](https://www.macports.org/):

    ```bash
    sudo port install graphviz
    ```

- **Windows**:

  - Download the installer from the [Graphviz Download Page](https://graphviz.org/download/).

  - Run the installer and ensure the option **"Add Graphviz to the system PATH for current user"** is selected during installation.

- **Linux (Debian/Ubuntu-based)**:

  - Install via APT: ([Linux Install Graphviz Dot - friendlylasopa](https://friendlylasopa340.weebly.com/linux-install-graphviz-dot.html?utm_source=chatgpt.com))

    ```bash
    sudo apt-get update
    sudo apt-get install graphviz
    ```

#### Verifying the Installation

After installation, confirm that the `dot` command is accessible:


```bash
dot -V
```

This should output the version of Graphviz installed, indicating that `dot` is ready for use.


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

Unit tests generate random task graphs and networks to verify scheduler correctness. They also check the RandomVariable utilities used for stochastic scheduling.

#### Locally

You can run the tests using `pytest`. Make sure you have installed the necessary dependencies, including `pytest` and `pytest-timeout`:

```bash
pip install pytest pytest-timeout
```

Then, run the tests:

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
### Examples

The repository contains several example scripts illustrating different algorithms and scenarios.
You can find them under [scripts/examples](./scripts/examples). To run an example, use:

```bash
python scripts/examples/<example_name>/main.py
```

The table of contents in `scripts/examples/Readme.md` lists examples ranging from basic usage to dynamic networks and scheduler comparisons.


### Experiments
To reproduce the experiments from papers using SAGA, see the [experiments](./scripts/experiments) directory.
Benchmarking utilities such as `exp_benchmarking.py` and `exp_parametric.py` help run full or batched experiments, and the PISA scripts include simulated annealing for adversarial instance generation.

### Workflow to get started
1. Select an example script or create your own workflow and network using the utilities in `saga.utils.random_graphs`.
2. Instantiate a scheduler from `saga.schedulers` and call its `schedule` method with your network and task graph.
3. Visualize the resulting schedule with functions like `saga.utils.draw.draw_gantt` or run experiments for systematic comparisons.
4. Use the unit tests or experiment scripts to benchmark new algorithms or reproduce results from the literature.

### Acknowledgements

This work was supported in part by Army Research Laboratory under Cooperative Agreement W911NF-17-2-0196.
