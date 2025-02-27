from functools import lru_cache
import pathlib
from typing import Iterable, Set, Tuple
from matplotlib import pyplot as plt
import pandas as pd
from itertools import combinations, product
import seaborn as sns

from exp_parametric import schedulers
from saga.utils.draw import gradient_heatmap

thisdir = pathlib.Path(__file__).resolve().parent

PARAM_NAMES = {
    'initial_priority': 'Priority',
    'append_only': 'Append Only',
    'compare': 'Compare',
    'critical_path': 'Critical Path',
    # 'k_depth': 'K Depth',
    'sufferage': 'Sufferage',
}
SCHEDULER_RENAMES = {
    "Insert": "Ins",
    "Append": "App",
    "Sufferage": "Suf",
    "ArbitraryTopological": "AT",
    "UpwardRanking": "UR",
    "CPoPRanking": "CR",
    "Quickest": "Quick",
    r"^EFT_Ins_CP_CR$": "CPoP",
    r"^EFT_Ins_UR$": "HEFT",
    r"^EFT_App_AT_Suf$": "Sufferage",
    r"^Quick_App_AT$": "MET",
    r"^EFT_App_AT$": "MCT",
}
@lru_cache(maxsize=None)
def load_data() -> pd.DataFrame:
    scheduler_params = {}
    for scheduler_name, scheduler in schedulers.items():
        details = scheduler.serialize()
        scheduler_params[scheduler_name] = {
            "initial_priority": details["initial_priority"]["name"],
            "append_only": details["insert_task"]["append_only"],
            "compare": details["insert_task"]["compare"],
            "critical_path": details["insert_task"].get("critical_path", False),
            "k_depth": details["k_depth"],
            "sufferage": 'sufferage_top_n' in details,
        }

    resultspath = thisdir / "results" / "parametric.csv"
    df = pd.read_csv(resultspath)

    for scheduler_name in df["scheduler"].unique():
        for key, value in scheduler_params[scheduler_name].items():
            df.loc[df["scheduler"] == scheduler_name, key] = value

    # clip runtimetime to 10
    df["runtime"] = df["runtime"].clip(lower=1/2)

    # Compute makespan ratio
    best_makespan = df.groupby(["dataset", "instance"]).agg({"makespan": "min"}).rename(columns={"makespan": "best_makespan"})
    df = df.join(best_makespan, on=["dataset", "instance"])
    df["makespan_ratio"] = df["makespan"] / df["best_makespan"]

    # Compute Runtime Ratio
    best_runtime = df.groupby(["dataset", "instance"]).agg({"runtime": "min"}).rename(columns={"runtime": "best_runtime"})
    df = df.join(best_runtime, on=["dataset", "instance"])
    df["runtime_ratio"] = df["runtime"] / df["best_runtime"]

    for key, value in SCHEDULER_RENAMES.items():
        df["scheduler"] = df["scheduler"].str.replace(key, value, regex=True)

    # assert there are no duplicate scheduler/data/instance combinations
    assert df.groupby(["scheduler", "dataset", "instance"]).size().max() == 1
    return df

def scheduler_table(df):
    df = df[["scheduler", *PARAM_NAMES.keys()]].drop_duplicates()
    df = df.sort_values(by=list(PARAM_NAMES.keys()))
    # df = df.set_index("scheduler")

    # Convert DataFrame to LaTeX table
    latex_table = df.to_latex(escape=True, index=False)
    savepath = thisdir / "output" / "parametric" / "scheduler_table.tex"
    savepath.parent.mkdir(parents=True, exist_ok=True)
    savepath.write_text(latex_table)

def get_missing_combos(df: pd.DataFrame) -> list[dict[str, str]]:
    param_values = {
        **{param: df[param].unique() for param in PARAM_NAMES},
        'dataset': df['dataset'].unique(),
    }
    missing_combos: list[dict[str, str]] = []
    for combo in product(*param_values.values()):
        combo = dict(zip(param_values.keys(), combo))
        if not df[(df[list(combo)] == pd.Series(combo)).all(axis=1)].empty:
            continue
        missing_combos.append(combo)
    return missing_combos

def print_scheduler_info():
    for scheduler_name, scheduler in schedulers.items():
        print(f"# {scheduler_name}")
        print(scheduler.serialize())
        print()

def print_data_info():
    df = load_data()
    print(df)

    missing_combos = get_missing_combos(df)
    print(f"Missing combinations: {len(missing_combos)}")

    missing_combos = get_missing_combos(df[df["k_depth"] <= 1])
    print(f"Missing combinations (k_depth <= 1): {len(missing_combos)}")

    missing_combos = get_missing_combos(df[df["k_depth"] == 0])
    print(f"Missing combinations (k_depth == 0): {len(missing_combos)}")

    missing_combos = get_missing_combos(df[df["dataset"] == "chains"])
    print(f"Missing combinations (dataset == chains): {len(missing_combos)}")

    missing_combos = get_missing_combos(df[df["sufferage"] == False])
    print(f"Missing combinations (sufferage == False): {len(missing_combos)}")

LABELS = {
    'makespan_ratio': 'Makespan Ratio',
    'runtime_ratio': 'Runtime Ratio',
    'initial_priority': 'Priority Function',
    'append_only': 'Append-Only Scheduling',
    'compare': 'Comparison Function',
    'critical_path': 'Critical Path Reservation',
    'ccr': 'CCR',
    'sufferage': 'Sufferage Consideration',
    'dataset_type': 'Dataset Type',
}

def generate_main_effect_plot(df: pd.DataFrame,
                              param_name: str,
                              metric: str,
                              savepath: pathlib.Path,
                              showfliers: bool = False):
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    df = df.copy()

    # rename UpwardRanking to UR, ArbitraryTopological to AT, and CriticalPath to CP
    if param_name == "initial_priority":
        df[param_name] = df[param_name].str.replace("UpwardRanking", "UR")
        df[param_name] = df[param_name].str.replace("ArbitraryTopological", "AT")
        df[param_name] = df[param_name].str.replace("CPoPRanking", "CR") 

    # Plotting
    # fig, ax = plt.subplots(figsize=(6, 6))
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(
        x=param_name, y=metric, data=df, ax=ax,
        showfliers=showfliers,
        # make color white
        boxprops=dict(facecolor=(1.0, 1.0, 1.0, 1.0))
    )
    ax.set_xlabel(LABELS[param_name])
    ax.set_ylabel(LABELS[metric])
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, bbox_inches='tight')
    print(f"Saved to {savepath}")
    plt.close()

def generate_interaction_plot(df: pd.DataFrame,
                              param_1: str,
                              param_2: str,
                              metric: str,
                              savepath: pathlib.Path):
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'h', '8', '*', 'H', 'd', 'X']
    linestyles = ["-", "--", "-.", ":", (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5))]

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.pointplot(
        x=param_1, y=metric, hue=param_2,
        data=df, dodge=True, ax=ax,
        markers=markers[:len(df[param_2].unique())],
        linestyles=linestyles[:len(df[param_2].unique())]
    )
    ax.set_xlabel(LABELS[param_1])
    ax.set_ylabel(LABELS[metric])
    # set legend title
    ax.legend(title=LABELS[param_2])

    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, bbox_inches='tight')
    print(f"Saved to {savepath}")
    plt.close()

def generate_interaction_plots(df: pd.DataFrame,
                               param_names: Iterable[str],
                               savedir: pathlib.Path,
                               showfliers: bool = False,
                               filetype: str = "pdf"):
    for param in param_names:
        generate_main_effect_plot(
            df, param, "makespan_ratio",
            savedir / f"{param}-makespan-ratio.{filetype}",
            showfliers=showfliers
        )
        generate_main_effect_plot(
            df, param, "runtime_ratio",
            savedir / f"{param}-runtime-ratio.{filetype}",
            showfliers=showfliers
        )

    for param_1, param_2 in combinations(param_names, 2):
        generate_interaction_plot(
            df, param_1, param_2, "makespan_ratio",
            savedir / "interactions" / f"{param_1}-{param_2}-makespan-ratio.{filetype}"
        )
        generate_interaction_plot(
            df, param_1, param_2, "runtime_ratio",
            savedir / "interactions" / f"{param_1}-{param_2}-runtime-ratio.{filetype}"
        )

def generate_pareto_front_plot(df: pd.DataFrame,
                               savedir: pathlib.Path,
                               varx: str = "dataset_type",
                               vary: str = "ccr",
                               filetype: str = "pdf",
                               figsize_chart: Tuple[int, int] = (16, 10),
                               figsize_scatter: Tuple[int, int] = (14, 12)):
    """Generate scatter plot of runtime_ratio vs makespan_ratio with pareto front highlighted

    Args:
        df (pd.DataFrame): DataFrame with makespan_ratio, runtime_ratio, scheduler, dataset, and ccr columns
        savedir (pathlib.Path): Directory to save the plot
        varx (str, optional): subplot variable on x-axis. Defaults to "ccr".
        vary (str, optional): subplot variable on y-axis. Defaults to "dataset".
        filetype (str, optional): Filetype to save the plot. Defaults to "pdf".
        figsize_chart (Tuple[int, int], optional): Size of the chart. Defaults to (16, 10).
        figsize_scatter (Tuple[int, int], optional): Size of the scatter plot. Defaults to (18, 10).

    """
    # aggregate makespan_ratio and runtime_ratio by scheduler
    df = df.groupby(by=["scheduler", "dataset", *PARAM_NAMES, varx, vary]).agg({"makespan_ratio": "mean", "runtime_ratio": "mean"}).reset_index()

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Highlight pareto front
    pareto_optimal_schedulers: Set[str] = set()
    varx_values = sorted(df[varx].unique())
    vary_values = sorted(df[vary].unique())
    for varx_value, vary_value in product(varx_values, vary_values):
        df_x = df[(df[varx] == varx_value) & (df[vary] == vary_value)]
        for scheduler in df_x["scheduler"].unique():
            df_scheduler = df_x[df_x["scheduler"] == scheduler]
            runtime_ratio_agg = df_scheduler["runtime_ratio"].values[0]
            makespan_ratio_agg = df_scheduler["makespan_ratio"].values[0]
            is_dominated = lambda rt1, mr1, rt2, mr2: (rt1 > rt2 and mr1 >= mr2) or (rt1 >= rt2 and mr1 > mr2)
            if not any(is_dominated(runtime_ratio_agg, makespan_ratio_agg, rt, mr) for rt, mr in zip(df_x["runtime_ratio"], df_x["makespan_ratio"])):
                pareto_optimal_schedulers.add(scheduler)
                df.loc[(df[varx] == varx_value) & (df[vary] == vary_value) & (df["scheduler"] == scheduler), "pareto"] = 1

    df = df[df["scheduler"].isin(pareto_optimal_schedulers)]
    scheduler_table(df)

    df = df.sort_values(by=["pareto"], ascending=[True], ignore_index=True, na_position="first")
    fig, ax = plt.subplots(len(vary_values), len(varx_values), figsize=figsize_scatter)
    for i, varx_value in enumerate(varx_values):
        for j, vary_value in enumerate(vary_values):
            df_subset = df[(df[varx] == varx_value) & (df[vary] == vary_value)]
            ax[j,i].scatter(
                df_subset["runtime_ratio"], df_subset["makespan_ratio"],
                c=df_subset["pareto"].apply(lambda x: "blue" if x == 1 else "red"),
                alpha=0.5
            )
            ax[j,i].set_title(f"{LABELS[vary]}={vary_value}, {LABELS[varx]}={varx_value}")
            if i == 0:
                ax[j,i].set_ylabel("Makespan Ratio")
            if j == len(vary_values) - 1:
                ax[j,i].set_xlabel("Runtime Ratio")
            
            # ax[j,i].set_xlabel("Runtime Ratio")
            # ax[j,i].set_ylabel("Makespan Ratio")

    plt.tight_layout()
    savepath = savedir / f"pareto_scatter.{filetype}"
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath)
    plt.close()

    df = df.sort_values(by=["pareto", "runtime_ratio"], ascending=[True, True], ignore_index=True)
    for varx_value, vary_value in product(varx_values, vary_values):
        df_idx = (df[varx] == varx_value) & (df[vary] == vary_value) & (df["pareto"] == 1)
        df.loc[df_idx, "pareto"] = df.loc[df_idx, "pareto"].cumsum()

    df = df.sort_values(by=[varx, vary, "pareto"], ascending=[True, True, True], ignore_index=True)
    ax = gradient_heatmap(
        df, x="scheduler", y="dataset",
        color="pareto",
        cmap="Blues",
        title="Pareto Optimal Schedule Makespan Ratio Rank",
        x_label="Scheduler",
        y_label="Dataset",
        color_label="Order (Runtime Ratio)",
        figsize=figsize_chart,
        cell_font_size=15,
        cmap_lower=0.2,
        cmap_upper=0.8,
    )

    savepath = savedir / f"pareto_chart.{filetype}"
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

    df.loc[df["pareto"].isna(), "makespan_ratio"] = None
    ax = gradient_heatmap(
        df, x="scheduler", y="dataset",
        color="makespan_ratio",
        cmap="coolwarm",
        upper_threshold=2,
        title="Pareto Optimal Schedules Makespan Ratio",
        x_label="Scheduler",
        y_label="Dataset",
        color_label="Makespan Ratio",
        figsize=(16, 10),
        cell_font_size=15,
    )

    savepath = savedir / f"pareto_chart_makespan_ratio.{filetype}"
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

    df.loc[df["pareto"].isna(), "runtime_ratio"] = None
    ax = gradient_heatmap(
        df, x="scheduler", y="dataset",
        color="runtime_ratio",
        cmap="coolwarm",
        upper_threshold=200,
        title="Pareto Optimal Schedules Runtime Ratio",
        x_label="Scheduler",
        y_label="Dataset",
        color_label="Runtime Ratio",
        figsize=(16, 10),
        cell_font_size=15,
    )

    savepath = savedir / f"pareto_chart_runtime_ratio.{filetype}"
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()


def gen_plots():
    filetype = "pdf"
    showfliers = False

    df = load_data()
    param_names = list(set(PARAM_NAMES) - {"k_depth"})
    df["ccr"] = df["dataset"].apply(lambda x: float(x.split('_ccr_')[1]))
    df["dataset_type"] = df["dataset"].apply(lambda x: x.split('_ccr_')[0])

    generate_pareto_front_plot(df, thisdir / "output" / "parametric", filetype=filetype)
    generate_interaction_plots(df, [*param_names, "dataset_type", "ccr"], thisdir / "output" / "parametric" , showfliers=showfliers, filetype=filetype)
    for dataset in df["dataset"].unique():
        print(f"Generating interaction plots for {dataset}")
        dataset_df = df[df["dataset"] == dataset]
        generate_interaction_plots(dataset_df, param_names, thisdir / "output" / "parametric" / "dataset" / dataset, showfliers=showfliers, filetype=filetype)

def main():
    # print_scheduler_info()
    # print_data_info()
    gen_plots()

if __name__ == '__main__':
    main()