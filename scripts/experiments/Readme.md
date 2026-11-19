# Experiments
Reproducible experiments for papers using SAGA.

## Benchmarking Experiments
See [benchmarking experiments](./benchmarking). Runs the scheduler suite over generated and wfcommons-derived datasets. Takes a few minutes.

## PISA Experiments
See [PISA experiments](./pisa). Searches for adversarial problem instances via simulated annealing. The full run takes an hour or more, so start with `run.py --quick`.

## Throughput Experiments
See [throughput experiments](./throughput_experiment). Compares realized throughput across a grid of schedulers and rescheduling policies, over deterministic and stochastic regimes. Run it as `uv run python run.py <branch> <regime>`, for example `uv run python run.py riotbench deterministic`.
</content>
