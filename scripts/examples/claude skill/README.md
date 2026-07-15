# agentic-hypothesis

Find broad families of scheduling problems where one [SAGA](https://github.com/ANRGUSC/saga) scheduler dramatically outperforms another. Given two algorithms, the workflow uses PISA (simulated annealing) to discover concrete adversarial instances, generalizes them into a parameterized random instance generator, and benchmarks the makespan-ratio distribution over many samples.

Results live under `outputs/<WINNER>_vs_<LOSER>/` (generator, benchmark report, plots). See [`outputs/FastestNode_vs_HEFT/report.md`](outputs/FastestNode_vs_HEFT/report.md) for a worked example (FastestNode beats HEFT by a geomean ~3.6x).

## Setup

```bash
git clone --recurse-submodules https://github.com/kubishi/agentic-hypothesis.git
# already cloned? git submodule update --init
```

Create the SAGA venv (used by all scripts as `saga/.venv/bin/python`) per the instructions in the `saga/` submodule.

## Using the skill with Claude Code

The `find-scheduler-family` skill lives in `.claude/skills/`. From Claude Code in this repo, invoke it with a `WINNER LOSER` pair:

```
/find-scheduler-family FastestNode HEFT
```

Claude then runs the full loop (hypothesize, PISA discovery, build the family generator, benchmark, iterate) and writes the report and plots to `outputs/`. Works for any pair in the SAGA registry, both directions (e.g. `HEFT CPoP`, `MinMin MaxMin`, `MET MCT`).
