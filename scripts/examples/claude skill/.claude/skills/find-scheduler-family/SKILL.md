---
name: find-scheduler-family
description: Find a broad family of problem instances where one SAGA scheduling algorithm dramatically outperforms another. Given two algorithm names (e.g. HEFT vs FastestNode), use PISA, structural analysis, and benchmarking to produce a parameterized random instance generator whose expected makespan ratio between the two algorithms is large. Use when asked to find adversarial families, distinguish schedulers, or characterize where one algorithm beats another.
argument-hint: "[WINNER] vs [LOSER]  (e.g. HEFT vs FastestNode)"
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Find a scheduler-adversarial instance family

## Goal

Given two SAGA schedulers, **WINNER** (should get low makespan) and **LOSER** (should
get high makespan), produce a **family**: a function

```python
def make_instance(rng: random.Random) -> tuple[Network, TaskGraph]: ...
```

that draws random problem instances such that the **expected** makespan of LOSER is much
larger than WINNER's. The deliverable is this generator plus a validation report showing
the makespan-ratio distribution over many samples, not a single hand-picked instance.

"Dramatically outperforms" is measured by the ratio `makespan(LOSER) / makespan(WINNER)`.
Target: geometric-mean ratio well above 1 (default threshold 2.0) AND consistent (p10
ratio comfortably above 1, so it's a broad family, not a lucky tail).

## Setup

- Run all Python with the SAGA venv: `saga/.venv/bin/python` (from the project root).
- Read [reference/saga_api.md](reference/saga_api.md) FIRST — it has the exact `Network.create` /
  `TaskGraph.create` contracts, the valid scheduler names, the homogeneity constraints,
  the CCR knob, and one-line intuitions for what each algorithm does. Getting the API
  contracts right (undirected network, add each edge pair once, super source/sink) saves
  a debugging loop.
- If the user names a pair but not which is WINNER/LOSER, pick the direction you expect to
  hold (usually the more sophisticated algorithm wins) and say so; the ratio is symmetric
  to relabel if you guessed wrong.

## Outputs

Everything is organized under `./outputs/<WINNER>_vs_<LOSER>/` in the project (the scripts
default there; run them from the project root). A completed run contains:

```
outputs/HEFT_vs_FastestNode/
├── report.md                 human-readable summary: hypothesis, ratio table, embedded plots
├── stats.json                machine-readable stats + raw ratios
├── family.py                 the exact generator that produced this report (provenance)
├── ratio_hist.png            distribution of makespan ratios (mean/median/threshold marked)
├── exemplar_task_graph.png   a representative instance's DAG
├── exemplar_network.png      its network (node speeds, link bandwidths)
├── exemplar_gantt.png        WINNER vs LOSER schedules stacked — the visual "why"
├── ccr_sweep.png             (if --ccr-sweep) adversarial gap vs CCR
└── seeds/                    best PISA-discovered instance: JSON + plots + structure summary
```

Open `report.md` first; it links every plot. Keep the final family generator as
`outputs/<WINNER>_vs_<LOSER>/family.py`.

## Workflow

Run scripts from the project root so outputs land in `./outputs`. Track progress with a
todo list.

### 1. Hypothesize from first principles
Before running anything, from the algorithm intuitions in the reference, write down WHY
LOSER should have a blind spot WINNER exploits (e.g. "FastestNode ignores parallelism").
This guides both the PISA seed and the family design.

### 2. Discover concrete adversarial instances with PISA
PISA runs simulated annealing that mutates a small instance to maximize
`makespan(LOSER)/makespan(WINNER)`. It reveals the structure that produces the gap.

```bash
saga/.venv/bin/python .claude/skills/find-scheduler-family/scripts/seed_pisa.py \
    --winner HEFT --loser FastestNode --restarts 6 --iterations 400
# saves JSON + plots + summary to ./outputs/HEFT_vs_FastestNode/seeds/
```

Read the printed structural summary (num tasks/nodes, level widths, weight ranges, CCR,
node/edge speed spread). If the best ratio is near 1.0, the direction may be wrong or the
gap only appears for larger/structured instances: try swapping WINNER/LOSER, `--init chain`
vs `branching`, more `--nodes`, or more iterations. The saved `*_summary.json` is your
blueprint.

PISA's mutation set only reweights/rewires an existing graph (add/delete dependency,
reweight task/dependency/node/edge) — it cannot add or remove tasks or nodes. The task/node
count is fixed by `--init`/`--nodes` for the whole search. If the adversarial structure you're
hypothesizing needs more or fewer tasks than the init provides, no amount of
`--iterations`/`--restarts` will find it — rerun with a different `--init`/size instead.

### 3. Generalize the seed into a family
**Start by perturbing the seed, not by freely re-randomizing it.** `*_summary.json` reports
each parameter's *marginal* range across the one discovered instance (e.g. "node speeds span
0.10-0.61") — that is not a safe sampling distribution. Independently redrawing each
parameter across its own marginal range usually destroys the specific *joint* alignment
(which rank/EFT comparison wins, which task contends with which) that made the seed
adversarial, even though every individual draw is "in range". This is the single most common
way a real gap silently collapses to NO EFFECT.

Instead, run `family_lib.sweep_perturbation` on the exact seed instance first:

```python
import json, random
from saga import Network, TaskGraph
from family_lib import sweep_perturbation, format_perturbation_sweep

net = Network(**json.load(open("outputs/<WINNER>_vs_<LOSER>/seeds/<tag>_network.json")))
tg = TaskGraph(**json.load(open("outputs/<WINNER>_vs_<LOSER>/seeds/<tag>_taskgraph.json")))
results = sweep_perturbation(net, tg, WINNER, LOSER, random.Random(0))
print(format_perturbation_sweep(results))
```

Read off where geomean/p10 hold up vs where they collapse:
- **Survives out past ~50-100% frac** → a *categorical* mechanism (the loser ignores some
  information altogether, e.g. FastestNode ignoring parallelism). Safe to build a broad,
  freely-restructured family per [templates/family_template.py](templates/family_template.py)
  Option A — vary node count, task count, weights, and topology within a band.
- **Only survives to ~10-20% frac** → a *comparison-fragile* mechanism (the gap depends on a
  delicate rank/EFT tie-break, e.g. HEFT vs CPoP). The family **is** `family_lib.perturb_instance`
  applied to the exact seed at a radius comfortably inside the observed cliff — copy the
  seed's exact values into `family.py` (Option B in the template) rather than re-deriving
  ranges by hand. This is a legitimate, reportable family, not a lesser result.
- **Collapses even at ~2% frac** → likely a numeric coincidence in the seed itself; see the
  NO EFFECT rule in step 4.

Set `WINNER`, `LOSER`, and update `HYPOTHESIS` either way.

Keep it a genuine family: a perturbation ball is one (uncountably many distinct instances) —
it does not need topology variation to count. A generator that emits one exact, unperturbed
instance is not a family and will be rejected in review.

Respect the homogeneity constraints from the reference if either algorithm needs them.

### 4. Benchmark the family
```bash
saga/.venv/bin/python .claude/skills/find-scheduler-family/scripts/benchmark_family.py \
    --family outputs/HEFT_vs_FastestNode/family.py --samples 300 --threshold 2.0 --ccr-sweep
# writes report.md + plots to ./outputs/HEFT_vs_FastestNode/ (add --stdout-only to skip files)
```

Read `outputs/<WINNER>_vs_<LOSER>/report.md`: mean/geomean/median ratio, p10/p90, fraction
above threshold, verdict, plus the ratio histogram and the WINNER-vs-LOSER Gantt that shows
the mechanism. `--exemplar max` visualizes the most dramatic instance instead of a median one.
- **STRONG** (geomean ≥ threshold and p10 ≥ 1.2): large AND consistent. Success.
- **MODERATE/CONSISTENT** (mean ≥ 1.2 and p10 ≥ 1.2): doesn't clear the threshold
  magnitude, but holds up even at the 10th percentile — a real, reliable, just smaller
  effect (e.g. geomean 1.9x with p10 1.9x too). Reportable as-is; step 5 can try to push
  the magnitude up, but this is not a failure.
- **WEAK/INCONSISTENT** (mean ≥ 1.2 but p10 < 1.2): there's a real average lean, but a
  meaningful fraction of samples don't show it (or even reverse) — go to step 5.
- **NO EFFECT** (mean < 1.2): rethink the hypothesis (step 1) or PISA seed (step 2).

p10, not stdev, is what separates MODERATE/CONSISTENT from WEAK/INCONSISTENT: it directly
answers "does the effect hold up for the unlucky tail," which is what matters for a broad
family — raw variance conflates a wide-but-always-winning distribution with a narrow-but-
sometimes-losing one. See `family_lib.classify_verdict` for the exact rule.

### 5. Iterate to widen and strengthen
Tune the generator to raise the geomean and lift the low tail (p10). Effective levers:
- **Sweep CCR** with `network.scale_to_ccr(task_graph, target_ccr=...)`. Many gaps only
  open at very low or very high CCR. Test a few values, keep the best band.
- Adjust the structural knob the hypothesis hinges on (DAG width, node-speed spread,
  dependency sizes, task-count range).
- Re-run PISA with different init/size if you're stuck for new structural ideas.
Re-benchmark after each change. Stop when the verdict is STRONG and stable across a couple
of seeds (`--seed 0`, `--seed 1`).

### 6. Deliver
Report to the user:
1. The final family file path and its `make_instance` signature.
2. The **hypothesis / mechanism** in plain language (why LOSER fails, why WINNER wins).
3. The benchmark table (geomean, median, p10/p90, fraction above threshold, sample size).
4. Which knobs (esp. CCR) control the effect, and any constraints/caveats (e.g. the family
   is only adversarial in a given CCR band, or requires ≥ N parallel tasks).

## Notes
- Ratios use makespan, which is deterministic per instance (schedulers are deterministic),
  so all randomness comes from `make_instance` drawing the instance. Always thread `rng`.
- `scripts/family_lib.py` holds the shared helpers (`resolve_scheduler`, `summarize_instance`,
  `evaluate_family`, `format_report`); import from it if you write custom analysis.
- This works for ANY pair in the registry, both directions. The same loop finds families
  for HEFT vs CPoP, MinMin vs MaxMin, MET vs MCT, etc.
