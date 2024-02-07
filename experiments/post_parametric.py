import pathlib
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.express as px
from sympy import Symbol, symbols, expand, log, S, Mul, Add, Poly
import pandas as pd

from exp_parametric import schedulers

thisdir = pathlib.Path(__file__).resolve().parent


def find_asymptotic_expr(expr, vars):
    expanded_expr = expand(expr)
    terms = expanded_expr.as_coefficients_dict().keys()

    # Function to evaluate the effective degree of a term
    # Polynomials get their total degree, log gets a special treatment
    def eval_degree(term, var):
        if term.has(log):
            # Remove log part and evaluate the degree of the rest
            return None

        return Poly(term, var).total_degree() if term.has(var) else S(0)
    
    all_degrees = {}
    for term in terms:
        degrees = {}  # degrees of each variable in the term
        for var in vars:
            degrees[var] = eval_degree(term, var)
        all_degrees[term] = degrees

    dominated_terms = set()
    for term, degrees in all_degrees.items():
        if term.has(log):
            continue
        for other_term, other_degrees in all_degrees.items():
            if term == other_term or other_term.has(log):
                continue
            if all(degrees[var] >= other_degrees[var] for var in vars):
                dominated_terms.add(other_term)
    
    dominating_terms = [term for term in terms if term not in dominated_terms]
    asymptotic_expr = Add(*dominating_terms)
    return asymptotic_expr


PARAM_NAMES = ['initial_priority', 'update_priority', 'append_only', 'compare', 'critical_path', 'k_depth']
def load_data() -> pd.DataFrame:
    scheduler_params = {}
    for scheduler_name, scheduler in schedulers.items():
        details = scheduler.serialize()
        scheduler_params[scheduler_name] = {
            "initial_priority": details["initial_priority"]["name"],
            "update_priority": details["update_priority"]["name"],
            "append_only": details["insert_task"]["append_only"],
            "compare": details["insert_task"]["compare"],
            "critical_path": details["insert_task"].get("critical_path", False),
            "k_depth": details["k_depth"],
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

def print_scheduler_info():
    for scheduler_name, scheduler in schedulers.items():
        print(f"# {scheduler_name}")
        print(scheduler.serialize())
        n, m, p = symbols('n m p')
        print(find_asymptotic_expr(scheduler.runtime(m, p, n, n**2), [n, m, p]))
        print()

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
            'update_priority': 'update_priority_NoUpdate',
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
    ols_fit()
    # plot_results()

if __name__ == '__main__':
    main()