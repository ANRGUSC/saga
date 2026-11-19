"""
parse.py — Summarise Inspirit parameter sweep results from output_data.csv.

For each recipe, prints:
  - HEFT baseline (makespan, throughput)
  - Best Inspirit config by throughput
  - Best Inspirit config by makespan
  - Improvement ratio vs HEFT for each
"""

import pathlib
import pandas as pd

thisdir = pathlib.Path(__file__).parent.absolute()
CSV = thisdir / "outputs" / "output_data.csv"


def main() -> None:
    df = pd.read_csv(CSV)

    # One row per (recipe, threshold, delta_ready) combination — drop per-step duplicates.
    combos = df.drop_duplicates(subset=["recipe", "threshold", "delta_ready"])[[
        "recipe", "num_tasks", "num_workers",
        "heft_makespan", "heft_throughput",
        "threshold", "delta_ready",
        "inspirit_makespan", "inspirit_throughput",
    ]]

    for recipe, group in combos.groupby("recipe"):
        row0 = group.iloc[0]
        heft_makespan = row0["heft_makespan"]
        heft_throughput = row0["heft_throughput"]

        best_tp = group.loc[group["inspirit_throughput"].idxmax()]
        best_ms = group.loc[group["inspirit_makespan"].idxmin()]

        print(f"\n{'=' * 60}")
        print(f"Recipe: {recipe}  ({int(row0['num_tasks'])} tasks, {int(row0['num_workers'])} workers)")
        print(f"{'=' * 60}")
        print(f"  HEFT baseline:   makespan={heft_makespan:.4f}   throughput={heft_throughput:.6f}")

        print(f"\n  Best throughput  (threshold={int(best_tp['threshold'])}, delta_ready={int(best_tp['delta_ready'])}):")
        print(f"    makespan={best_tp['inspirit_makespan']:.4f}  ({_pct(best_tp['inspirit_makespan'], heft_makespan):+.1f}% vs HEFT)")
        print(f"    throughput={best_tp['inspirit_throughput']:.6f}  ({_pct(best_tp['inspirit_throughput'], heft_throughput):+.1f}% vs HEFT)")

        print(f"\n  Best makespan    (threshold={int(best_ms['threshold'])}, delta_ready={int(best_ms['delta_ready'])}):")
        print(f"    makespan={best_ms['inspirit_makespan']:.4f}  ({_pct(best_ms['inspirit_makespan'], heft_makespan):+.1f}% vs HEFT)")
        print(f"    throughput={best_ms['inspirit_throughput']:.6f}  ({_pct(best_ms['inspirit_throughput'], heft_throughput):+.1f}% vs HEFT)")

        print(f"\n  All combinations (sorted by throughput desc):")
        ranked = group.sort_values("inspirit_throughput", ascending=False).reset_index(drop=True)
        print(f"  {'#':>3}  {'threshold':>9}  {'delta_ready':>11}  {'makespan':>12}  {'throughput':>12}")
        print(f"  {'-'*3}  {'-'*9}  {'-'*11}  {'-'*12}  {'-'*12}")
        for i, r in ranked.iterrows():
            print(
                f"  {i+1:>3}  {int(r['threshold']):>9}  {int(r['delta_ready']):>11}"
                f"  {r['inspirit_makespan']:>12.4f}  {r['inspirit_throughput']:>12.6f}"
            )


def _pct(val: float, baseline: float) -> float:
    if baseline == 0:
        return float("nan")
    return (val - baseline) / baseline * 100


if __name__ == "__main__":
    main()
