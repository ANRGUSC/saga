import pathlib
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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

    resultsdir = thisdir / "results" / "parametric"
    dfs = []
    for data in resultsdir.glob("*/*.csv"):
        dataset_name = data.stem
        scheduler_name = data.parent.stem
        df = pd.read_csv(data, index_col=0)
        # df.drop(columns=["scheduler"], inplace=True)
        df["dataset"] = dataset_name
        for param, value in scheduler_params[scheduler_name].items():
            df[param] = value
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    
    df.to_csv(thisdir / "results" / "parametric" / "makespan_ratio.csv", index=False)
    
    best_makespan = df.groupby(["dataset", "instance"]).agg({"makespan": "min"}).rename(columns={"makespan": "best_makespan"})
    df = df.join(best_makespan, on=["dataset", "instance"])
    df["makespan_ratio"] = df["makespan"] / df["best_makespan"]
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
        print(df_with_dummies.columns)
        ref_categories = {
            'initial_priority': 'initial_priority_ArbitraryTopological',
            'append_only': 'append_only_True',  # Assuming the original value is a boolean, adjust if it's actually a string
            'compare': 'compare_EST',
            'critical_path': 'critical_path_False',
            "k_depth": "k_depth_0",
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

        print(f"# {dataset}")
        print(model.summary())
        print()

def plot_runtime_by_makespan_ratio():
    """Plot runtime by makespan ratio for each scheduler/dataset
    
    The marker size represents the variance of the makespan ratio for each scheduler
    """

    df = load_data()
    df = df.groupby(by=["scheduler", "dataset"]).agg({"runtime": "max", "makespan_ratio": ["mean", "std"]}).reset_index()
    df.columns = ["scheduler", "dataset", "runtime", "makespan_ratio_mean", "makespan_ratio_std"]
    
    # get max across datasets
    df = df.groupby(by=["scheduler"]).agg({"runtime": "max", "makespan_ratio_mean": "max", "makespan_ratio_std": "max"}).reset_index()
    print(df)

    fig = px.scatter(
        df,
        x="runtime",
        y="makespan_ratio_mean",
        color="scheduler",
        # symbol="dataset",
        size="makespan_ratio_std",
        title="Runtime by Makespan Ratio",
        template="plotly_white",
    )
    fig.write_html(thisdir / "output" / "parametric" / "runtime_by_makespan_ratio.html")


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