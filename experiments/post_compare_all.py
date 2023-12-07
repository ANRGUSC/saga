import pathlib

import matplotlib

matplotlib.use("Agg")

from functools import lru_cache
from typing import Dict

import dill as pickle
import matplotlib.pyplot as plt
import pandas as pd
from plot import gradient_heatmap
from simulated_annealing import SimulatedAnnealing

thisdir = pathlib.Path(__file__).parent.absolute()

@lru_cache(maxsize=1)
def load_results(resultspath: pathlib.Path) -> Dict[str, Dict[str, SimulatedAnnealing]]:
    results = {}
    for base_path in resultspath.glob("*"):
        results[base_path.name] = {}
        for path in base_path.glob("*.pkl"):
            results[base_path.name][path.stem] = pickle.loads(path.read_bytes())
    return results


def to_df(results: Dict[str, Dict[str, SimulatedAnnealing]]) -> pd.DataFrame:
    rows = []
    for base_scheduler_name, base_scheduler_results in results.items():
        for scheduler_name, scheduler_results in base_scheduler_results.items():
            makespan_ratio = scheduler_results.iterations[-1].best_energy
            rows.append([base_scheduler_name, scheduler_name, makespan_ratio])

    df_results = pd.DataFrame(rows, columns=["Base Scheduler", "Scheduler", "Makespan Ratio"])
    return df_results

def load_results_csv(outputpath: pathlib.Path) -> pd.DataFrame:
    df_results = pd.read_csv(outputpath.joinpath("results.csv"), index_col=0)
    return df_results

def results_to_csv(resultspath: pathlib.Path,
                   outputpath: pathlib.Path):
    df_results = to_df(load_results(resultspath))
    df_results.to_csv(outputpath.joinpath("results.csv"))

def print_stats(outputpath: pathlib.Path):
    df_results = load_results_csv(outputpath)
    df_hybrid = df_results[df_results["Scheduler"].str.startswith("Not")]
    for row in df_hybrid.itertuples():
        print(row)


def tab_results(resultsdir: pathlib.Path,
                savedir: pathlib.Path,
                upper_threshold: float = 5.0,
                include_hybrid = False,
                add_worst_row = True) -> None:
    """Generate table of results."""
    savedir.mkdir(parents=True, exist_ok=True)
    df_all_results = load_results_csv(resultsdir)

    # rename some schedulers via dict
    rename_dict = {
        "CPOP": "CPoP",
        "Fastest Node": "FastestNode",
    }
    rename_dict = {
        **rename_dict,
        **{f"Not{key}": f"Not{value}" for key, value in rename_dict.items()}
    }
    df_all_results["Scheduler"] = df_all_results["Scheduler"].replace(rename_dict)
    df_all_results["Base Scheduler"] = df_all_results["Base Scheduler"].replace(rename_dict)

    df_results = df_all_results[
        (~df_all_results["Scheduler"].str.startswith("Not")) & 
        (~df_all_results["Base Scheduler"].str.startswith("Not"))]
    if include_hybrid:
        hybrid_values = []
        for scheduler in df_results["Scheduler"].unique():
            # get NotScheduler Base Scheduler result
            res = df_all_results[
                (df_all_results["Scheduler"] == scheduler) &
                (df_all_results["Base Scheduler"] == f"Not{scheduler}")
            ]
            try:
                hybrid_values.append([scheduler, "Hybrid", res["Makespan Ratio"].values[0]])
            except IndexError:
                pass

        # append hybrid values to df_results
        df_hybrid = pd.DataFrame(hybrid_values, columns=["Scheduler", "Base Scheduler", "Makespan Ratio"])
        df_results = pd.concat([df_results, df_hybrid], ignore_index=True)

    if add_worst_row:
        worst_results = df_results.groupby("Scheduler")["Makespan Ratio"].max()
        df_worst = pd.DataFrame(
            [[scheduler, "Worst", worst_results[scheduler]]
             for scheduler in worst_results.index],
            columns=["Scheduler", "Base Scheduler", "Makespan Ratio"]
        )
        df_results = pd.concat([df_results, df_worst], ignore_index=True)

    axis = gradient_heatmap(
        df_results,
        x="Scheduler",
        y="Base Scheduler",
        color="Makespan Ratio",
        upper_threshold=upper_threshold,
        x_label="Scheduler",
        y_label="Base Scheduler",
        color_label="Makespan Ratio",
        # custom order so that "Hybrid" and "Worst" are at the bottom
        xorder=lambda x: x.replace("Hybrid", "ZHybrid").replace("Worst", "ZWorst"),
        yorder=lambda y: y.replace("Hybrid", "ZHybrid").replace("Worst", "ZWorst")
    )
    plt.tight_layout()
    axis.get_figure().savefig(
        savedir / "results.pdf",
        dpi=300,
        bbox_inches='tight'
    )
    axis.get_figure().savefig(
        savedir / "results.png",
        dpi=300,
        bbox_inches='tight'
    )

