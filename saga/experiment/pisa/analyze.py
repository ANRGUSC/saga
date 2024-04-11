import pathlib
from functools import lru_cache
from typing import Dict

import dill as pickle
import matplotlib.pyplot as plt
import pandas as pd
from saga.experiment.plot import gradient_heatmap
from simulated_annealing import SimulatedAnnealing
from saga.experiment.benchmarking.analyze import SCHEDULER_RENAMES

thisdir = pathlib.Path(__file__).parent.absolute()

@lru_cache(maxsize=1)
def load_results(resultspath: pathlib.Path) -> Dict[str, Dict[str, SimulatedAnnealing]]:
    """Load results from resultspath.

    Args:
        resultspath: path to results directory

    Returns:
        results: dict of dicts of SimulatedAnnealing objects
    """
    results = {}
    for base_path in resultspath.glob("*"):
        results[base_path.name] = {}
        for path in base_path.glob("*.pkl"):
            results[base_path.name][path.stem] = pickle.loads(path.read_bytes())
    return results


def to_df(results: Dict[str, Dict[str, SimulatedAnnealing]]) -> pd.DataFrame:
    """Convert results to dataframe.

    Args:
        results: dict of dicts of SimulatedAnnealing objects

    Returns:
        df_results: dataframe of results
    """
    rows = []
    for base_scheduler_name, base_scheduler_results in results.items():
        for scheduler_name, scheduler_results in base_scheduler_results.items():
            makespan_ratio = scheduler_results.iterations[-1].best_energy
            rows.append([base_scheduler_name, scheduler_name, makespan_ratio])

    df_results = pd.DataFrame(rows, columns=["Base Scheduler", "Scheduler", "Makespan Ratio"])
    return df_results

def load_results_csv(outputpath: pathlib.Path) -> pd.DataFrame:
    """Load results from outputpath.

    Args:
        outputpath: path to output directory

    Returns:
        df_results: dataframe of results
    """
    df_results = pd.read_csv(outputpath.joinpath("results.csv"), index_col=0)
    return df_results

def results_to_csv(resultspath: pathlib.Path,
                   outputpath: pathlib.Path):
    """Convert results to csv.

    Args:
        resultspath: path to results directory
        outputpath: path to output directory

    Returns:
        df_results: dataframe of results
    """
    df_results = to_df(load_results(resultspath))
    df_results.to_csv(outputpath.joinpath("results.csv"))

def tab_results(resultsdir: pathlib.Path,
                savedir: pathlib.Path,
                upper_threshold: float = 5.0,
                include_hybrid = False,
                add_worst_row = True,
                title: str = None,
                savename: str  = "results",
                mode: str = None) -> None:
    """Generate table of results.

    Args:
        resultsdir: path to results directory
        savedir: path to save directory
        upper_threshold: upper threshold for heatmap
        include_hybrid: whether to include hybrid results
        add_worst_row: whether to add a row for the worst result
        title: title for plot
        savename: name for plot
        mode: "pdf", "png", or None. None saves both.
    """
    savedir.mkdir(parents=True, exist_ok=True)
    df_all_results = load_results_csv(resultsdir)

    # rename some schedulers via dict
    rename_dict = {
        "CPOP": "CPoP",
        "Fastest Node": "FastestNode",
        **SCHEDULER_RENAMES
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

    def default_order(x):
        return x.replace("Hybrid", "ZHybrid").replace("Worst", "ZWorst").replace(r"\textit", "AA")

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
        xorder=default_order,
        yorder=default_order,
        # include_cell_labels=True,
        title=title,
        cell_font_size=12.0
    )
    plt.tight_layout()
    if mode is None or mode == "pdf":
        axis.get_figure().savefig(
            savedir / f"{savename}.pdf",
            dpi=300,
            bbox_inches='tight'
        )
    if mode is None or mode == "png":
        axis.get_figure().savefig(
            savedir / f"{savename}.png",
            dpi=300,
            bbox_inches='tight'
        )

