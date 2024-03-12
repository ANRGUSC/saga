import pathlib
import dill as pickle
from functools import lru_cache
from typing import Dict

import pandas as pd
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
