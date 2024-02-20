import json
from math import factorial
import pathlib
from typing import Iterable, Set
from matplotlib import pyplot as plt
from statsmodels.multivariate.manova import MANOVA
import pandas as pd
from itertools import combinations, product
import seaborn as sns

from exp_parametric import schedulers

thisdir = pathlib.Path(__file__).resolve().parent

PARAM_NAMES = ['initial_priority', 'append_only', 'compare', 'critical_path', 'k_depth', 'sufferage']
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
    df = pd.read_csv(resultspath, index_col=0)

    for scheduler_name in df["scheduler"].unique():
        for key, value in scheduler_params[scheduler_name].items():
            df.loc[df["scheduler"] == scheduler_name, key] = value
    
    # Compute makespan ratio
    best_makespan = df.groupby(["dataset", "instance"]).agg({"makespan": "min"}).rename(columns={"makespan": "best_makespan"})
    df = df.join(best_makespan, on=["dataset", "instance"])
    df["makespan_ratio"] = df["makespan"] / df["best_makespan"]

    # Compute Runtime Ratio
    best_runtime = df.groupby(["dataset", "instance"]).agg({"runtime": "min"}).rename(columns={"runtime": "best_runtime"})
    df = df.join(best_runtime, on=["dataset", "instance"])
    df["runtime_ratio"] = df["runtime"] / df["best_runtime"]
    return df

def get_missing_combos(df: pd.DataFrame) -> list[dict[str, str]]:
    param_values = {
        **{param: df[param].unique() for param in PARAM_NAMES},
        'dataset': df['dataset'].unique(),
    }
    missing_combos: list[dict[str, str]] = []
    for combo in product(*param_values.values()):
        combo = dict(zip(param_values.keys(), combo))
        # print("combo", combo)
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

    # Check if there are combinations of parameter values that are not present in the data
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

def generate_interaction_plots(df: pd.DataFrame,
                               param_names: Iterable[str],
                               savedir: pathlib.Path,
                               showfliers: bool = False):
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'h', '8', '*', 'H', 'd', 'X']
    linestyles = ["-", "--", "-.", ":", (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5))]
    for param in param_names:
        print(f"Plotting main effects for {param}")
        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(18, 10))

        # Boxplot for makespan_ratio by Compare
        sns.boxplot(
            x=param, y='makespan_ratio', data=df, ax=axs[0],
            showfliers=showfliers
        )
        axs[0].set_title(f'Makespan Ratio by {param}')

        # Boxplot for runtime_ratio by Compare
        sns.boxplot(
            x=param, y='runtime_ratio', data=df, ax=axs[1],
            showfliers=showfliers
        )
        axs[1].set_title(f'Runtime Ratio by {param}')

        plt.tight_layout()
        savepath = savedir / "main_effects" / f"{param}.png"
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath)
        print(f"Saved to {savepath}")

        plt.close()

    for param_1, param_2 in combinations(param_names, 2):
        # sort param_1 and param_2 so that the plot is consistent
        param_1, param_2 = sorted([param_1, param_2])
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))

        sns.boxplot(x=param_1, y='makespan_ratio', data=df, ax=axs[0, 0], showfliers=showfliers)
        axs[0, 0].set_title(f'Makespan Ratio by {param_1}')
        sns.boxplot(x=param_1, y='runtime_ratio', data=df, ax=axs[0, 1], showfliers=showfliers)
        axs[0, 1].set_title(f'Runtime Ratio by {param_1}')

        sns.boxplot(x=param_2, y='makespan_ratio', data=df, ax=axs[1, 0], showfliers=showfliers)
        axs[1, 0].set_title(f'Makespan Ratio by {param_2}')
        sns.boxplot(x=param_2, y='runtime_ratio', data=df, ax=axs[1, 1], showfliers=showfliers)
        axs[1, 1].set_title(f'Runtime Ratio by {param_2}')

        # Interaction Plots
        sns.pointplot(
            x=param_1, y='makespan_ratio', hue=param_2,
            data=df, dodge=True, 
            markers=markers[:len(df[param_2].unique())],
            linestyles=linestyles[:len(df[param_2].unique())],
            ax=axs[2, 0]
        )
        axs[2, 0].set_title(f'Interaction: Makespan Ratio by {param_1} and {param_2}')
        sns.pointplot(
            x=param_1, y='runtime_ratio', hue=param_2,
            data=df, dodge=True,
            markers=markers[:len(df[param_2].unique())],
            linestyles=linestyles[:len(df[param_2].unique())],
            ax=axs[2, 1]
        )
        axs[2, 1].set_title(f'Interaction: Runtime Ratio by {param_1} and {param_2}')

        plt.tight_layout()
        savepath = savedir / "interactions" / f"{param_1}:{param_2}.png"
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath)
        plt.close()

def generate_pareto_front_plot(df: pd.DataFrame,
                               savedir: pathlib.Path,
                               varx: str = "ccr",
                               vary: str = "dataset_name"):
    """Generate scatter plot of runtime_ratio vs makespan_ratio with pareto front highlighted

    Args:
        df (pd.DataFrame): DataFrame with makespan_ratio, runtime_ratio, scheduler, dataset, and ccr columns
        savedir (pathlib.Path): Directory to save the plot
        varx (str, optional): subplot variable on x-axis. Defaults to "ccr".
        vary (str, optional): subplot variable on y-axis. Defaults to "dataset".
    
    """
    # aggregate makespan_ratio and runtime_ratio by scheduler
    df = df.groupby(by=["scheduler", *PARAM_NAMES, varx, vary]).agg({"makespan_ratio": "mean", "runtime_ratio": "mean"}).reset_index()


    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Highlight pareto front
    pareto_optimal_schedules: Set[str] = set()
    for varx_values, vary_values in df[[varx, vary]].drop_duplicates().values:
        df_x = df[(df[varx] == varx_values) & (df[vary] == vary_values)]
        for scheduler in df_x["scheduler"].unique():
            df_scheduler = df_x[df_x["scheduler"] == scheduler]
            runtime_ratio_agg = df_scheduler["runtime_ratio"].values[0]
            makespan_ratio_agg = df_scheduler["makespan_ratio"].values[0]
            is_dominated = lambda rt1, mr1, rt2, mr2: (rt1 > rt2 and mr1 >= mr2) or (rt1 >= rt2 and mr1 > mr2)
            if not any(is_dominated(runtime_ratio_agg, makespan_ratio_agg, rt, mr) for rt, mr in zip(df_x["runtime_ratio"], df_x["makespan_ratio"])):
                pareto_optimal_schedules.add(scheduler)
                df.loc[(df[varx] == varx_values) & (df[vary] == vary_values) & (df["scheduler"] == scheduler), "pareto"] = True

    df.loc[df["pareto"] != True, "pareto"] = False

    # Sort dataframe by [varx, vary, "pareto"]
    df = df.sort_values(by=["pareto"], ascending=[True], ignore_index=True)
    
    # drop all rows where scheduler is not in pareto_optimal_schedules
    df = df[df["scheduler"].isin(pareto_optimal_schedules)]
    print(df[[*PARAM_NAMES]].drop_duplicates(ignore_index=True))

    fig, ax = plt.subplots(len(df[vary].unique()), len(df[varx].unique()), figsize=(18, 10))
    for i, varx_value in enumerate(df[varx].unique()):
        for j, vary_value in enumerate(df[vary].unique()):
            df_subset = df[(df[varx] == varx_value) & (df[vary] == vary_value)]
            ax[j,i].scatter(
                df_subset["runtime_ratio"], df_subset["makespan_ratio"],
                c=df_subset["pareto"].apply(lambda x: "blue" if x else "red"),
                alpha=0.5
            )
            ax[j,i].set_title(f"{vary}={vary_value}, {varx}={varx_value}")
            ax[j,i].set_xlabel("Runtime Ratio")
            ax[j,i].set_ylabel("Makespan Ratio")

    plt.tight_layout()
    savepath = savedir / "pareto_front.png"
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath)
    plt.close()
    

def anova():
    df = load_data()
    # MANOVA Analysis
    formula = 'makespan_ratio + runtime_ratio ~ initial_priority'
    manova = MANOVA.from_formula(formula, data=df)
    savepath = thisdir / "output" / "parametric" / "anova" / "summary.txt"
    savepath.parent.mkdir(parents=True, exist_ok=True)
    savepath.write_text(manova.mv_test().summary().as_text())

def interactions():
    df = load_data()
    param_names = list(set(PARAM_NAMES) - {"k_depth"})
    # df = df[["dataset", *param_names, "makespan_ratio", "runtime_ratio"]]
    df["ccr"] = df["dataset"].apply(lambda x: float(x.split('_ccr_')[1]))
    df["dataset_name"] = df["dataset"].apply(lambda x: x.split('_ccr_')[0])

    generate_interaction_plots(df, [*param_names, "dataset_name", "ccr"], thisdir / "output" / "parametric" , showfliers=False)
    generate_pareto_front_plot(df, thisdir / "output" / "parametric")
    for dataset in df["dataset"].unique():
        print(f"Generating interaction plots for {dataset}")
        dataset_df = df[df["dataset"] == dataset]
        generate_interaction_plots(dataset_df, param_names, thisdir / "output" / "parametric" / "dataset" / dataset, showfliers=False)

def main():
    # print_scheduler_info()
    # print_data_info()
    # anova()
    interactions()

if __name__ == '__main__':
    main()