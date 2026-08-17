# PISA Experiments

These reproduce the PISA experiments from the paper [Comparing Task Graph Scheduling Algorithms: An Adversarial Approach](https://arxiv.org/abs/2403.07120). PISA uses simulated annealing to search for problem instances where one scheduler performs badly relative to another.

Run all commands from this directory.

## Quick Run

Start here if you just want to see the pipeline work:

```bash
uv run python run.py --quick   # 4 schedulers, 2 tries each: about a minute
uv run python analyze.py       # Analyze the results
```

## Full Run

The full experiment covers every ordered pair of the 16 schedulers with 10 random restarts each:

```bash
uv run python run.py       # Run the experiments
uv run python analyze.py   # Analyze the results
```

Be aware of what this costs before starting it: about **an hour** on a fast desktop and **several hours** on a small machine such as a 2-core Codespace, and roughly **1GB of disk**. It is single-process, so it does not benefit from extra cores. Results are written per pair and `run.py` skips pairs that already have results, so you can stop it with Ctrl-C and resume later by re-running the same command.

Between those two extremes, you can scale the experiment however you like:

```bash
uv run python run.py --schedulers HEFT CPoP OLB MinMin --num-tries 3
uv run python run.py --max-iterations 300
```

See `uv run python run.py --help` for all options.

## Results

`run.py` saves the best run for each scheduler pair in `./results`, and `analyze.py` generates `./output/results.csv` and the heatmap `./output/results.png`. The heatmap only shows the pairs you actually ran, so a partial run produces a partial table.

While it runs, each try keeps a working directory under `./results/.runs`; these are deleted as each pair finishes. Pass `--keep-runs` to retain them for inspection, but note that they take about 15x the space of the results themselves (~15GB for a full run).

You can also open the `./explore_results.ipynb` Jupyter notebook to interactively explore the results for different pairs of scheduling algorithms. Select the `.venv` interpreter as the notebook kernel.
</content>
