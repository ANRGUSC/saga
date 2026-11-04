# Priority-Function Experiment

How much does the task-ordering (priority) function of a list scheduler matter? This experiment holds the insertion policy fixed and swaps only the initial priority, using `ParametricScheduler` so the baseline and its randomized counterpart differ in nothing else.

| Name | Initial priority | Insertion | Samples per instance |
| --- | --- | --- | --- |
| `HEFT` | Upward rank | EFT insertion | 1 |
| `HEFT_RandomOrder` | Random topological order | EFT insertion | 100 |
| `CPoP` | CPoP rank | EFT insertion, critical path pinned to fastest node | 1 |
| `CPoP_RandomOrder` | Random topological order | EFT insertion, critical path pinned to fastest node | 100 |
| `RandomOrder_RandomNode` | Random topological order | Random allowed node | 100 |

The random order is Kahn's algorithm with a uniformly random pick from the ready set (`RandomTopological` in `saga.schedulers.parametric.components`), so every valid topological order has positive probability. Each sample uses a seed derived from the dataset, instance, and sample index, so results are reproducible and `run.py` can be interrupted and resumed at sample granularity. Each row of the results carries `Sample` and `Seed` columns.

Every instance's network is scaled to a target communication-to-computation ratio (`target_ccr` in `common.py`, default 1.0) with `Network.scale_to_ccr`, which rescales all inter-node links by one factor and so preserves their relative heterogeneity. This matters most for the riotbench fog networks, whose unscaled CCR is in the hundreds: there every list scheduler puts all tasks on the single cloud node, the baseline makespan is nearly zero, and ratios against it are meaningless. Random-network node and link speeds are floored at 0.1 because the library default clips at 1e-9, which occasionally yields a dead link that dominates the makespan.

`analyze.py` compares each randomized scheduler against the baseline that shares its insertion policy and writes a per-dataset summary (median, best, and worst ratio of random-order makespan to baseline makespan, and the fraction of random orders that beat the baseline), a per-instance table, and boxplots of the ratio per dataset on a log scale.

Run from this directory:
```bash
uv run python prepare.py # Prepares the datasets
uv run python run.py # Runs the experiments
uv run python analyze.py # Analyzes the results, producing plots
```
Datasets, results, and plots go in `./data`, `./results`, and `./output`. Schedulers and sample counts are configured in `common.py`.
