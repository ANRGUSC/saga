"""Evaluate whether task_score predicts good duplication targets.

For each random instance, rank tasks by task_score, then measure makespan when
duplicating the top-N vs the bottom-N tasks (N = 0..max). If the heuristic is
useful, duplicating top-N tasks should reduce makespan more (or hurt less) than
duplicating bottom-N tasks.

The duplication decision is injected by monkeypatching should_duplicate in the
HEFT and CPOP modules to allow only a chosen set of tasks; the schedulers are not
modified.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import saga.schedulers.cpop as cpop_mod
import saga.schedulers.heft as heft_mod
from heuristics import task_score
from saga import Network, TaskGraph
from saga.schedulers import CpopScheduler, HeftScheduler
from saga.utils.random_graphs import get_branching_dag, get_network

thisdir = pathlib.Path(__file__).parent.resolve()
outdir = thisdir / "output"

NUM_EXPERIMENTS = 100
MAX_TOP_N = 5
DUPLICATION_FACTOR = 2

# Tasks whose duplication the patched should_duplicate will allow this iteration.
allowed_duplicates: set[str] = set()


def select_should_duplicate(
    task_name: str, task_graph: TaskGraph, network: Network
) -> bool:
    return task_name in allowed_duplicates


def ranked_tasks(task_graph: TaskGraph, mode: str) -> list[str]:
    """Real tasks ranked by task_score, highest first for 'top-n', lowest for 'bottom-n'."""
    scored = [
        (task.name, task_score(task.name, task_graph))
        for task in task_graph.tasks
        if not task.name.startswith("__super_")
    ]
    scored.sort(key=lambda item: item[1], reverse=(mode == "top-n"))
    return [name for name, _ in scored]


def main() -> None:
    global allowed_duplicates
    np.random.seed(0)

    heft_mod.should_duplicate = select_should_duplicate
    cpop_mod.should_duplicate = select_should_duplicate

    schedulers = {"heft": HeftScheduler, "cpop": CpopScheduler}

    rows = []
    for experiment in range(NUM_EXPERIMENTS):
        network = get_network()
        task_graph = get_branching_dag()
        for scheduler_name, scheduler_class in schedulers.items():
            for mode in ("top-n", "bottom-n"):
                ranking = ranked_tasks(task_graph, mode)
                for n in range(MAX_TOP_N + 1):
                    allowed_duplicates = set(ranking[:n])
                    schedule = scheduler_class(
                        duplication_factor=DUPLICATION_FACTOR
                    ).schedule(network, task_graph)
                    rows.append(
                        {
                            "experiment": experiment,
                            "scheduler": scheduler_name,
                            "mode": mode,
                            "n": n,
                            "makespan": schedule.makespan,
                        }
                    )

    df = pd.DataFrame(rows)
    outdir.mkdir(exist_ok=True)
    df.to_csv(outdir / "data.csv", index=False)

    for scheduler_name in df["scheduler"].unique():
        summary = (
            df[df["scheduler"] == scheduler_name]
            .groupby(["mode", "n"])["makespan"]
            .agg(["mean", "std"])
            .reset_index()
        )
        print(f"\n{scheduler_name.upper()}")
        print(summary.to_string(index=False))

        plt.figure(figsize=(8, 5))
        for mode in summary["mode"].unique():
            mode_df = summary[summary["mode"] == mode]
            plt.errorbar(
                mode_df["n"],
                mode_df["mean"],
                yerr=mode_df["std"],
                marker="o",
                capsize=5,
                label=mode,
            )
        plt.xlabel("number of duplicated tasks")
        plt.ylabel("average makespan")
        plt.title(f"Top-N vs bottom-N duplication ({scheduler_name.upper()})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / f"{scheduler_name}_makespan_plot.png")
        plt.close()
    print(f"\nwrote results to {outdir}")


if __name__ == "__main__":
    main()
