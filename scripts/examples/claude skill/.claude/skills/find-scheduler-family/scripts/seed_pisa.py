"""Discover concrete adversarial instances for a scheduler pair via PISA.

PISA (simulated annealing) mutates a small instance to maximize
makespan(loser) / makespan(winner). Running several restarts surfaces the
structural features that make `winner` beat `loser` — the raw material for a
generalized family. This script prints, for the best instance found, a
structural summary (see family_lib.summarize_instance) and saves the instance
JSON so you can inspect or replay it.

Usage:
    python seed_pisa.py --winner HEFT --loser FastestNode \
        --restarts 6 --iterations 400 --out ./seeds

Direction: --winner is the algorithm you expect to WIN (lower makespan).
PISA searches for instances where --loser does much worse.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from family_lib import summarize_instance, save_instance_figures  # noqa: E402

from saga.pisa import SimulatedAnnealing, SimulatedAnnealingConfig  # noqa: E402
from saga.utils.random_graphs import get_branching_dag, get_chain_dag, get_network  # noqa: E402
from saga.utils.random_variable import RandomVariable, UniformRandomVariable  # noqa: E402

# Schedulers requiring a homogeneous dimension (see scripts/experiments/pisa/run.py).
HOMOGENOUS_COMP = {"ETF", "FCP", "FLB"}
HOMOGENOUS_COMM = {"BIL", "GDL", "FCP", "FLB"}

ALL_CHANGES = [
    "TaskGraphAddDependency",
    "TaskGraphDeleteDependency",
    "TaskGraphChangeDependencyWeight",
    "TaskGraphChangeTaskWeight",
    "NetworkChangeEdgeWeight",
    "NetworkChangeNodeWeight",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--winner", required=True, help="Scheduler expected to win (lower makespan).")
    p.add_argument("--loser", required=True, help="Scheduler expected to lose.")
    p.add_argument("--restarts", type=int, default=6, help="Random restarts (best kept).")
    p.add_argument("--iterations", type=int, default=400, help="SA iterations per restart.")
    p.add_argument("--init", choices=["branching", "chain"], default="branching")
    p.add_argument("--nodes", type=int, default=4, help="Initial network node count.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, default=None,
                   help="Dir to save best instance JSON. Default: ./outputs/<winner>_vs_<loser>/seeds.")
    args = p.parse_args()
    if args.out is None:
        args.out = pathlib.Path("outputs") / f"{args.winner}_vs_{args.loser}" / "seeds"

    random.seed(args.seed)
    data_dir = pathlib.Path(tempfile.mkdtemp(prefix="pisa_seed_"))

    node_dist = UniformRandomVariable(0.1, 1.0)
    edge_dist = UniformRandomVariable(0.1, 1.0)
    changes = list(ALL_CHANGES)
    if args.winner in HOMOGENOUS_COMP or args.loser in HOMOGENOUS_COMP:
        node_dist = RandomVariable(samples=[1.0])
        changes.remove("NetworkChangeNodeWeight")
    if args.winner in HOMOGENOUS_COMM or args.loser in HOMOGENOUS_COMM:
        edge_dist = RandomVariable(samples=[1.0])
        changes.remove("NetworkChangeEdgeWeight")

    best_energy = 0.0
    best_iter = None
    for r in range(args.restarts):
        net = get_network(args.nodes, node_weight_distribution=node_dist, edge_weight_distribution=edge_dist)
        if args.init == "chain":
            tg = get_chain_dag(4)
        else:
            tg = get_branching_dag(levels=3, branching_factor=2,
                                   node_weight_distribution=UniformRandomVariable(0.1, 1.0),
                                   edge_weight_distribution=UniformRandomVariable(0.1, 1.0))
        sa = SimulatedAnnealing(
            name=f"{args.winner}_vs_{args.loser}_r{r}",
            scheduler=args.loser,       # energy = loser.makespan / winner.makespan
            base_scheduler=args.winner,
            initial_network=net,
            initial_task_graph=tg,
            config=SimulatedAnnealingConfig(
                max_iterations=args.iterations, max_temp=10.0, min_temp=0.1,
                cooling_rate=0.985, change_types=changes,
            ),
            data_dir=data_dir,
        )
        sa.execute(progress=False)
        it = sa.best_iteration
        print(f"  restart {r}: energy (ratio loser/winner) = {it.current_energy:.3f}")
        if it.current_energy > best_energy:
            best_energy = it.current_energy
            best_iter = it

    if best_iter is None:
        print("No instance found.")
        return

    net = best_iter.current_network
    tg = best_iter.current_task_graph
    summary = summarize_instance(net, tg)
    print("\n=== BEST INSTANCE ===")
    print(f"ratio loser/winner (makespan): {best_energy:.3f}")
    print("structure:")
    print(json.dumps(summary, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.winner}_vs_{args.loser}"
    (args.out / f"{tag}_network.json").write_text(net.model_dump_json(indent=2))
    (args.out / f"{tag}_taskgraph.json").write_text(tg.model_dump_json(indent=2))
    (args.out / f"{tag}_summary.json").write_text(json.dumps(
        {"ratio": best_energy, "summary": summary}, indent=2))
    save_instance_figures(net, tg, args.winner, args.loser, args.out)
    print(f"\nSaved best instance (JSON + plots) to {args.out}/")


if __name__ == "__main__":
    main()
