import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from plot import gradient_heatmap
from post_load import load_results_csv

thisdir = pathlib.Path(__file__).parent.absolute()

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

