import pathlib
from typing import Set
import statsmodels.api as sm
import numpy as np
import plotly.express as px
import pandas as pd
from itertools import product

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
    df = pd.read_csv(resultspath)

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


def ols_fit():
    df = load_data()
    for dataset in df["dataset"].unique():
        df_dataset = df[df["dataset"] == dataset]
        df_dataset = df_dataset[[*PARAM_NAMES, "makespan_ratio"]]
        df_dataset = df_dataset.groupby(by=PARAM_NAMES).mean().reset_index()

        df_with_dummies = pd.get_dummies(df_dataset, columns=PARAM_NAMES, drop_first=False)
        ref_categories = {
            'initial_priority': 'initial_priority_ArbitraryTopological',
            'append_only': 'append_only_True',  # Assuming the original value is a boolean, adjust if it's actually a string
            'compare': 'compare_EST',
            'critical_path': 'critical_path_False',
            'sufferage': 'sufferage_False',
        }
        excluded_vars = list(ref_categories.values())

        # Interactions
        # df_with_dummies['sufferageXcritical'] = df_with_dummies['update_priority_SufferageUpdatePriority'] * df_with_dummies['critical_path_True']
        # df_with_dummies['EFTxInsert'] = df_with_dummies['compare_EFT'] * df_with_dummies['append_only_False']
        # df_with_dummies['HEFT'] = df_with_dummies['compare_EFT'] * df_with_dummies['append_only_False'] * df_with_dummies['initial_priority_UpwardRanking']

        X = df_with_dummies.drop(excluded_vars, axis=1).drop(columns=["makespan_ratio"])
        y = df_with_dummies['makespan_ratio']
        X = sm.add_constant(X)
        model = sm.OLS(y, X.astype(float)).fit()

        savepath = thisdir / "output" / "parametric" / "ols" / f"{dataset}.txt"
        savepath.parent.mkdir(parents=True, exist_ok=True)
        savepath.write_text(model.summary().as_text())

def plot_runtime_by_makespan_ratio():
    """Plot runtime by makespan ratio for each scheduler/dataset
    
    The marker size represents the variance of the makespan ratio for each scheduler
    """

    df = load_data()
    # do .split('_ccr_')[0] to get dataset name
    df["dataset"] = df["dataset"].apply(lambda x: x.split('_ccr_')[0])

    df = df.groupby(by=["scheduler", "dataset", "ccr"]).agg({"runtime_ratio": "max", "makespan_ratio": ["max"]}).reset_index()
    df.columns = ["scheduler", "dataset", "ccr", "runtime_ratio_max", "makespan_ratio_max"]
    
    # Keep only schedulers that are pareto-optimal in at least one dataset/ccr combination
    # pareto optimality w.r.t. runtime_ratio_max and makespan_ratio_max
    df["pareto"] = False
    pareto_optimal_schedules: Set[str] = set()
    for dataset, ccr in df[["dataset", "ccr"]].drop_duplicates().values:
        df_dataset = df[(df["dataset"] == dataset) & (df["ccr"] == ccr)]
        for scheduler in df_dataset["scheduler"].unique():
            df_scheduler = df_dataset[df_dataset["scheduler"] == scheduler]
            runtime_ratio_max = df_scheduler["runtime_ratio_max"].values[0]
            makespan_ratio_max = df_scheduler["makespan_ratio_max"].values[0]
            is_dominated = lambda rt1, mr1, rt2, mr2: (rt1 > rt2 and mr1 >= mr2) or (rt1 >= rt2 and mr1 > mr2)
            if not any(is_dominated(runtime_ratio_max, makespan_ratio_max, rt, mr) for rt, mr in zip(df_dataset["runtime_ratio_max"], df_dataset["makespan_ratio_max"])):
                # set "pareto" column to True for scheduler/dataset/ccr combo in original df
                df.loc[(df["dataset"] == dataset) & (df["ccr"] == ccr) & (df["scheduler"] == scheduler), "pareto"] = True
                pareto_optimal_schedules.add(scheduler)

    df = df[df["scheduler"].isin(pareto_optimal_schedules)]    
    df['marker_size'] = df['pareto'].apply(lambda x: 5 if x else 1)
    fig = px.scatter(
        df,
        x="runtime_ratio_max",
        y="makespan_ratio_max",
        color="scheduler",
        facet_col="ccr",
        facet_row="dataset",
        size="marker_size",
        title="Runtime by Makespan Ratio",
        template="plotly_white",
        hover_data=["scheduler", "dataset", "ccr", "runtime_ratio_max"],
    )
    fig.write_html(thisdir / "output" / "parametric" / "runtime_by_makespan_ratio.html")
    print(f"Saved to {thisdir / 'output' / 'parametric' / 'runtime_by_makespan_ratio.html'}")


def plot_results():
    savedir = thisdir / "output" / "parametric"
    savedir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    for dataset in df["dataset"].unique():
        # get stats for each scheduler makespan ratio
        # desc = df[["scheduler", "makespan_ratio"]].groupby(by=["scheduler"]).describe()
        df_dataset = df[df["dataset"] == dataset]
        desc = df_dataset[["scheduler", "makespan_ratio"]].groupby(by=["scheduler"]).describe()
        # print(desc)

        # print stats for standard deviation (mean, std, etc.)
        # I want to know the std of the std
        # print(desc["makespan_ratio"]["std"].describe())

        # sort by makespan ratio and then plot 
        df_dataset = df_dataset.sort_values(by="makespan_ratio")
        x = np.arange(len(df_dataset))
        
        fig = px.scatter(
            df_dataset,
            x=x,
            y="makespan_ratio",
            color="critical_path",
            symbol="append_only",
            facet_col="initial_priority",
            facet_row="update_priority",
            title=f"{dataset} Makespan Ratio",
            template="plotly_white",
        )
        
        fig.write_html(savedir / f"{dataset}.html")

def main():
    # print_scheduler_info()
    # print_data_info()
    # ols_fit()
    # plot_results()
    plot_runtime_by_makespan_ratio()

if __name__ == '__main__':
    main()