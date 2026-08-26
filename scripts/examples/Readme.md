# Examples
Standalone examples of how to use the library, different use-cases, etc.

Run any example with:

```bash
uv run python scripts/examples/<example_name>/main.py
```

Most examples write their plots to an `outputs/` directory next to the script.

## Table of Contents
- [Example 1: Basic Example](./basic_example) — a manual schedule, the optimal schedule, and HEFT, side by side.
- [Example 2: Parametric Scheduler](./basic_example_parametric) — build your own scheduler by mixing and matching algorithmic components.
- [Example 3: Stochastic Scheduling](./basic_example_stochastic) — task and communication times as random variables instead of fixed values.

## Additional Examples

These are not part of the guided walkthrough, but are useful references:

- [Online Scheduling](./basic_example_online) — scheduling as tasks arrive, using `OnlineParametricScheduler`.
- [Throughput Scheduling](./basic_example_throughput) — optimizing throughput rather than makespan, comparing `MultiObjScheduler` against HEFT and CPoP.
- [Online Environment](./online_example_environment) — driving the `Environment` simulation loop with the Inspirit policy.
- [Online Environment (large)](./online_example_environment_big) — a parameter sweep of the above across wfcommons recipes, writing `outputs/output_data.csv`. This one takes a long time to run; `parse.py` summarizes the committed CSV without re-running it.
- [FrontierHEFT vs FIFO](./frontier_heft_vs_fifo) — comparing two online schedulers on a Montage workflow.
- [Classic Structured DAGs](./classic_dags) — Gaussian elimination and FFT task graphs from the HEFT/CPoP paper, comparing several schedulers on each.
</content>
</invoke>
