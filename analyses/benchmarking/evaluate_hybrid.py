import pathlib
import pandas as pd
from analyze import load_data
from datasets import load_dataset
from saga.schedulers import (
    BILScheduler, CpopScheduler, DuplexScheduler,
    ETFScheduler, FastestNodeScheduler, FCPScheduler,
    GDLScheduler, HeftScheduler, MaxMinScheduler,
    MinMinScheduler, MCTScheduler, METScheduler,
    OLBScheduler, WBAScheduler, FLBScheduler
)
from saga.schedulers.hybrid import HybridScheduler
thisdir = pathlib.Path(__file__).parent.absolute()

TRAIN_DATASETS = [
    "in_trees",
    "chains",
    "out_trees",
    "blast",
    "epigenomics", "genome", "seismology",
    "soykb",
    "bwa",
    "cycles", "montage",
    "srasearch",


]
EVAL_DATASETS = [
    "etl", "predict",
    "stats", "train"
]

SCHEDULERS = {
    "CPOP": CpopScheduler(),
    "HEFT": HeftScheduler(),
    "Duplex": DuplexScheduler(),
    "ETF": ETFScheduler(),
    "FastestNode": FastestNodeScheduler(),
    "FCP": FCPScheduler(),
    "GDL": GDLScheduler(),
    "MaxMin": MaxMinScheduler(),
    "MinMin": MinMinScheduler(),
    "MCT": MCTScheduler(),
    "MET": METScheduler(),
    "OLB": OLBScheduler(),
    "BIL": BILScheduler(),
    "WBA": WBAScheduler(),
    "FLB": FLBScheduler(),
}

r_scheduler_names = {
    sched.__class__.__name__: name
    for name, sched in SCHEDULERS.items()
}

hybrid_adversarial = [
    'MinMin', 'MET', 'FCP',
    'WBA', 'HEFT', 'Duplex',
    'OLB', 'BIL', 'CPOP',
    'MCT', 'ETF', 'FastestNode',
    'GDL', 'MaxMin', 'FLB'
]

def main():
    df_results = load_data()

    # rename values in scheduler column
    df_results["scheduler"] = df_results["scheduler"].replace(r_scheduler_names)

    # add "instance" column that numbers each row for each dataset/scheduler (in current order)
    df_results["instance"] = df_results.groupby(["dataset", "scheduler"]).cumcount()

    # keep only datasets in TRAIN_DATASETS
    df_results = df_results[df_results["dataset"].isin(TRAIN_DATASETS)]

    df_agg_dataset = df_results.groupby(["scheduler", "dataset"])["makespan_ratio"].mean()
    df_agg = df_agg_dataset.groupby("scheduler").mean().sort_values(ascending=True)
    hybrid_bench_mean = df_agg.index.tolist()
    
    df_agg_dataset = df_results.groupby(["scheduler", "dataset"])["makespan_ratio"].mean()
    df_agg = df_agg_dataset.groupby("scheduler").min().sort_values(ascending=True)
    hybrid_bench_min_mean = df_agg.index.tolist()

    df_agg_dataset = df_results.groupby(["scheduler", "dataset"])["makespan_ratio"].max()
    df_agg = df_agg_dataset.groupby("scheduler").min().sort_values(ascending=True)
    hybrid_bench_min_max = df_agg.index.tolist()

    hybrid_strategies = {
        "Adversarial": hybrid_adversarial,
        "BenchMean": hybrid_bench_mean,
        "BenchMinMean": hybrid_bench_min_mean,
        "BenchMinMax": hybrid_bench_min_max,
    }

    # print strategies
    for stra_name, stra in hybrid_strategies.items():
        print(f"{stra_name}: {stra}")

    max_num_algs = len(hybrid_strategies["Adversarial"])

    for dataset_name in EVAL_DATASETS:
        for strat_name, hybrid_strat in hybrid_strategies.items():
            df_results_hybrid = None
            savepath = thisdir.joinpath("results_hybrid", dataset_name, f"{strat_name}.csv")
            savepath.parent.mkdir(exist_ok=True, parents=True)
            for i in range(max_num_algs):
                _df = df_results[df_results["scheduler"].isin(hybrid_strat[:i+1])]
                df_hybrid = _df.groupby(["dataset", "instance"])["makespan_ratio"].min().reset_index()
                df_hybrid["scheduler"] = f"{strat_name}.{i}"

                if df_results_hybrid is None:
                    df_results_hybrid = df_hybrid
                else:
                    df_results_hybrid = pd.concat([df_results_hybrid, df_hybrid], ignore_index=True)

                df_results_hybrid.to_csv(savepath)

if __name__ == "__main__":
    main()