"""Measure the expected makespan ratio of a candidate family, with plots.

Given a Python file exposing `make_instance(rng: random.Random) -> (Network, TaskGraph)`,
this samples many instances and reports the distribution of
makespan(loser) / makespan(winner). Use it to validate that a family reliably
(not just occasionally) makes `winner` beat `loser`.

Usage:
    # text report to stdout only:
    python benchmark_family.py --family ./my_family.py \
        --winner HEFT --loser FastestNode --samples 300 --threshold 2.0

    # full visual report into a run directory:
    python benchmark_family.py --family ./my_family.py --samples 300 \
        --out ./runs/HEFT_vs_FastestNode [--ccr-sweep] [--exemplar median|max]

With --out you get, in that directory:
    report.md                 summary + verdict + embedded image links
    ratio_hist.png            distribution of makespan ratios
    exemplar_task_graph.png   a representative instance's DAG
    exemplar_network.png      its network
    exemplar_gantt.png        winner vs loser schedules side by side
    ccr_sweep.png             (with --ccr-sweep) gap vs CCR

The family file may optionally define `WINNER`, `LOSER`, and `HYPOTHESIS`
module constants; CLI flags override WINNER/LOSER.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from family_lib import (  # noqa: E402
    evaluate_family, format_report, save_ratio_histogram, classify_verdict,
    save_instance_figures, pick_exemplar, save_ccr_sweep, estimate_ccr,
)


def load_family(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("candidate_family", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "make_instance"):
        raise AttributeError(f"{path} must define make_instance(rng) -> (Network, TaskGraph)")
    return mod


def write_report(outdir, mod, stats, exemplar_info, ccr_means, family_path):
    lines = [
        f"# Family report: {stats['winner']} (winner) vs {stats['loser']} (loser)",
        "",
        f"Family source: `{family_path}`",
        "",
        "## Hypothesis",
        "",
        getattr(mod, "HYPOTHESIS", "_(none provided in family file)_"),
        "",
        "## Makespan ratio  loser / winner",
        "",
        "| metric | value |",
        "|---|---|",
        f"| samples usable | {stats['usable']} / {stats['n']} ({stats['errors']} errors) |",
        f"| geomean | {stats['geomean_ratio']:.3f} |",
        f"| mean | {stats['mean_ratio']:.3f} |",
        f"| median | {stats['median_ratio']:.3f} |",
        f"| stdev | {stats['stdev_ratio']:.3f} |",
        f"| p10 / p90 | {stats['p10_ratio']:.3f} / {stats['p90_ratio']:.3f} |",
        f"| min / max | {stats['min_ratio']:.3f} / {stats['max_ratio']:.3f} |",
        f"| frac ≥ {stats['threshold']} | {stats['frac_above_threshold']:.1%} |",
        f"| mean makespan winner / loser | {stats['mean_winner_makespan']:.3f} / {stats['mean_loser_makespan']:.3f} |",
        f"| **verdict** | **{classify_verdict(stats)}** |",
        "",
        "![ratio histogram](ratio_hist.png)",
        "",
    ]
    if exemplar_info:
        lines += [
            "## Exemplar instance",
            "",
            f"Representative instance: winner makespan {exemplar_info['winner_makespan']:.3f}, "
            f"loser makespan {exemplar_info['loser_makespan']:.3f} (ratio {exemplar_info['ratio']:.3f}).",
            "",
            "![task graph](exemplar_task_graph.png)",
            "",
            "![network](exemplar_network.png)",
            "",
            "![gantt: winner vs loser](exemplar_gantt.png)",
            "",
        ]
    if ccr_means:
        best = max(ccr_means, key=lambda k: ccr_means[k])
        lines += [
            "## CCR sweep",
            "",
            f"Strongest gap at CCR ≈ {best:g} (geomean ratio {ccr_means[best]:.3f}).",
            "",
            "![ccr sweep](ccr_sweep.png)",
            "",
        ]
    cost_note = getattr(mod, "CLAUDE_COST_ESTIMATE", None)
    if cost_note:
        lines += [
            "## Claude API cost",
            "",
            cost_note,
            "",
        ]
    (pathlib.Path(outdir) / "report.md").write_text("\n".join(lines))
    # machine-readable stats too (drop the big ratios list into its own key)
    dump = {k: v for k, v in stats.items() if k != "ratios"}
    dump["ratios"] = stats["ratios"]
    (pathlib.Path(outdir) / "stats.json").write_text(json.dumps(dump, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, type=pathlib.Path, help="Path to family .py file.")
    p.add_argument("--winner", default=None, help="Scheduler expected to win (lower makespan).")
    p.add_argument("--loser", default=None, help="Scheduler expected to lose.")
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=2.0,
                   help="'Dramatic' ratio threshold for the frac-above and verdict.")
    p.add_argument("--out", type=pathlib.Path, default=None,
                   help="Directory for the visual report (report.md + PNGs). "
                        "Default: ./outputs/<winner>_vs_<loser>. Use --stdout-only to skip files.")
    p.add_argument("--stdout-only", action="store_true", help="Print the text report only; write no files.")
    p.add_argument("--exemplar", choices=["median", "max"], default="median",
                   help="Which sampled instance to visualize (default: median = representative).")
    p.add_argument("--ccr-sweep", action="store_true",
                   help="Also plot the adversarial gap across CCR values (needs comm > 0).")
    args = p.parse_args()

    mod = load_family(args.family)
    winner = args.winner or getattr(mod, "WINNER", None)
    loser = args.loser or getattr(mod, "LOSER", None)
    if not winner or not loser:
        p.error("Provide --winner/--loser or define WINNER/LOSER in the family file.")

    stats = evaluate_family(mod.make_instance, winner, loser, args.samples, random.Random(args.seed), args.threshold)
    print(format_report(stats))

    if args.stdout_only:
        return
    if not stats.get("usable"):
        print("No usable samples; skipping plots.")
        return
    if args.out is None:
        args.out = pathlib.Path("outputs") / f"{winner}_vs_{loser}"

    args.out.mkdir(parents=True, exist_ok=True)
    # Keep a copy of the exact family that produced this report (provenance).
    import shutil
    shutil.copyfile(args.family, args.out / "family.py")
    save_ratio_histogram(stats, args.out / "ratio_hist.png")

    exemplar_info = None
    ex = pick_exemplar(mod.make_instance, winner, loser, random.Random(args.seed + 1),
                       min(args.samples, 200), mode=args.exemplar)
    if ex is not None:
        net, tg, _ = ex
        exemplar_info = save_instance_figures(net, tg, winner, loser, args.out)

    ccr_means = None
    if args.ccr_sweep:
        # only meaningful if the family has communication
        probe_net, probe_tg = mod.make_instance(random.Random(args.seed + 2))
        if estimate_ccr(probe_net, probe_tg) > 0:
            ccr_means = save_ccr_sweep(mod.make_instance, winner, loser,
                                       random.Random(args.seed + 3),
                                       [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
                                       min(args.samples, 120), args.out / "ccr_sweep.png")
        else:
            print("CCR sweep skipped: family has zero communication (dependency sizes are 0).")

    write_report(args.out, mod, stats, exemplar_info, ccr_means, args.family)
    print(f"\nVisual report written to {args.out}/report.md")


if __name__ == "__main__":
    main()
