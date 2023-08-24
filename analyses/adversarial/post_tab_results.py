import pathlib
import pickle
from typing import Dict, Hashable, List

import networkx as nx
import pandas as pd
from saga.schedulers.base import Task
from saga.schedulers import CpopScheduler, HeftScheduler

from simulated_annealing import SimulatedAnnealing

thisdir = pathlib.Path(__file__).parent.absolute()

def load_results() -> Dict[str, Dict[str, SimulatedAnnealing]]:
    results = {}
    for base_path in (thisdir / "results").glob("*"):
        results[base_path.name] = {}
        for path in base_path.glob("*.pkl"):
            results[base_path.name][path.stem] = pickle.loads(path.read_bytes())
    return results

def main():
    keep_schedulers = [
        # "CPOP",
        # "HEFT",
        # "Duplex",
        # "MaxMin",
        # "MinMin",
        # # "FCP",
        # "MET",
        # "MCT",
    ]
    results = load_results()

    rows = []
    for base_scheduler_name, base_scheduler_results in results.items():
        if keep_schedulers and not base_scheduler_name in keep_schedulers:
            continue
        for scheduler_name, scheduler_results in base_scheduler_results.items():
            if keep_schedulers and not scheduler_name in keep_schedulers:
                continue
            makespan_ratio = scheduler_results.iterations[-1].best_energy
            if makespan_ratio > 1e3:
                makespan_ratio = ">1000"
            else:
                makespan_ratio = f"{makespan_ratio:.2f}"
            rows.append({
                "Base Scheduler": base_scheduler_name,
                "Scheduler": scheduler_name,
                "Makespan Ratio": makespan_ratio
            })

    df_results = pd.DataFrame(rows)
    # pivot so base schedulers are columns and schedulers are rows
    df_results = df_results.pivot(index="Scheduler", columns="Base Scheduler", values="Makespan Ratio")
    df_results = df_results.reindex(sorted(df_results.columns), axis=1)
    df_results = df_results.reindex(sorted(df_results.index), axis=0)

    df_results = df_results.fillna("")
    df_results = df_results.replace("nan", "")

    (thisdir / "output").mkdir(exist_ok=True)
    df_results.to_csv(thisdir / "output" / "results.csv")
    df_results.to_string(buf=thisdir / "output" / "results.txt")

    df_results.to_latex(
        buf=thisdir / "output" / "results.tex",
        escape=False,
        column_format="|c|c|c|c|c|c|c|c|c|c|c|c|c|c|",
        multicolumn_format="c",
        multicolumn=True,
        na_rep="N/A",
        bold_rows=True,
        caption="Makespan ratio for each scheduler relative to the base scheduler.",
        label="tab:sa_results"
    )

    df_results.to_html(buf=thisdir / "output" / "results.html")

if __name__ == "__main__":
    main()
