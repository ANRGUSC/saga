# Duplication heuristics experiment

Experimental scoring heuristics for choosing which tasks to duplicate, and an
evaluation of whether they predict good duplication targets. This is research
code; the heuristics are **not** wired into any SAGA scheduler.

- `heuristics.py` — candidate task-duplication scores (`communication_score`,
  `impact_score`, `branching_score`, `join_val_score`, and their mean `task_score`).
- `score_examples.py` — prints the scores on a few hand-built graphs as a sanity check.
- `run.py` — sweeps random instances, duplicating the top-N vs bottom-N scored
  tasks (via a monkeypatched `should_duplicate`) and compares makespan. Writes
  `output/data.csv` and per-scheduler plots.

Run from this directory:

```bash
uv run python score_examples.py
uv run python run.py
```

Outputs land in `output/` (git-ignored).
