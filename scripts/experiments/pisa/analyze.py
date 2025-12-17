import json
import logging
import pathlib
from functools import lru_cache

import matplotlib.pyplot as plt
import pandas as pd
from saga.utils.draw import gradient_heatmap


thisdir = pathlib.Path(__file__).parent.absolute()


SCHEDULER_RENAMES = {
    "Cpop": "CPoP",
    "Heft": "HEFT",
}

@lru_cache(maxsize=1)
def load_results(resultspath: pathlib.Path) -> pd.DataFrame:
    """Load results from resultspath.

    Args:
        resultspath: path to results directory

    Returns:
        df_results: dataframe of results
    """
    rows = []
    for run_dir in resultspath.glob("*_vs_*"):
        if not run_dir.is_dir():
            continue
        
        run_json_path = run_dir / "run.json"
        if not run_json_path.exists():
            logging.warning(f"No run.json found in {run_dir}")
            continue
        
        # Load the JSON data
        with open(run_json_path, 'r') as f:
            data = json.load(f)
        
        # Extract base_scheduler and scheduler from the directory name
        # Format: {base_scheduler}_vs_{scheduler}
        parts = run_dir.name.split("_vs_")
        if len(parts) == 2:
            base_scheduler = parts[0]
            scheduler = parts[1]
            makespan_ratio = data.get("best_energy", 0.0)
            rows.append([base_scheduler, scheduler, makespan_ratio])
    
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
    df_results = load_results(resultspath)
    df_results.to_csv(outputpath.joinpath("results.csv"))

def tab_results(resultsdir: pathlib.Path,
                savedir: pathlib.Path,
                upper_threshold: float = 5.0,
                include_hybrid = False,
                add_worst_row = True,
                title: str | None = None,
                savename: str  = "results",
                mode: str | None = None) -> None:
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

    def default_order(x: str) -> str:
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
        fig = axis.get_figure()
        if fig is not None:
            fig.savefig(
                savedir / f"{savename}.pdf",
                dpi=300,
                bbox_inches='tight'
            )
    if mode is None or mode == "png":
        fig = axis.get_figure()
        if fig is not None:
            fig.savefig(
                savedir / f"{savename}.png",
                dpi=300,
                bbox_inches='tight'
            )


def main():
    logging.basicConfig(level=logging.INFO)

    resultsdir = thisdir.joinpath("results")
    outputdir = thisdir.joinpath("output")

    outputdir.mkdir(parents=True, exist_ok=True)

    results_to_csv(resultsdir, outputdir)
    tab_results(outputdir, outputdir, mode="pdf")

if __name__ == "__main__":
    main()
