# Classic Structured DAGs

Generates the two classic structured task-graph families used to evaluate list schedulers in the heterogeneous-scheduling literature — **Gaussian elimination** and **FFT** — with the exact topologies specified in the HEFT/CPoP paper (Topcuoglu, Hariri and Wu, *IEEE TPDS* 13(3), 2002, [doi:10.1109/71.993206](https://doi.org/10.1109/71.993206), Figs. 8 and 10), and compares several SAGA schedulers on them. The same families appear in the evaluation of PEFT (Arabnejad and Barbosa, *IEEE TPDS* 25(3), 2014, [doi:10.1109/TPDS.2013.57](https://doi.org/10.1109/TPDS.2013.57)); the Gaussian elimination graph traces back to Wu and Gajski's Hypertool (1990) and Cosnard et al. (1988), and the FFT graph to Chung and Ranka (1992).

The generators verify the paper's closed-form task counts on every run: Gaussian elimination on an `m x m` matrix has `(m^2 + m - 2) / 2` tasks, and an `N`-point FFT has `2N - 1` recursive-call tasks plus `N*log2(N)` butterfly tasks. The FFT graph has `N` exit tasks; because SAGA requires a single exit task per graph, it adds a zero-cost `__super_sink__`. That dummy task is scheduled internally and never affects makespans. The drawings omit it and therefore match the figures in the papers.

Run with:

```bash
uv run python scripts/examples/classic_dags/main.py
```

The script prints a makespan comparison per instance and saves the task-graph drawings and HEFT Gantt charts to `outputs/`.
