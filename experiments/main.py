import argparse
import pathlib
from functools import partial
from typing import Dict

import pandas as pd

import exp_benchmarking
import exp_compare_all
import exp_wfcommons
import exp_stochastic_benchmarking
import post_benchmarking
import post_compare_all
import prepare_datasets
import prepare_stochastic_datasets
import logging

from saga.schedulers import (BILScheduler, CpopScheduler, DuplexScheduler,
                             ETFScheduler, FastestNodeScheduler, FCPScheduler,
                             FLBScheduler, GDLScheduler, HeftScheduler,
                             MaxMinScheduler, MCTScheduler, METScheduler,
                             MinMinScheduler, OLBScheduler, WBAScheduler)

thisdir = pathlib.Path(__file__).parent.absolute()
logging.basicConfig(level=logging.INFO)

def chain(funcs):
    """Chain multiple functions together."""
    def chained_func(*args, **kwargs):
        for func in funcs:
            func(*args, **kwargs)
    return chained_func

def tab_results_ccr(resultsdir: pathlib.Path,
                    savedir: pathlib.Path,
                    benchmarking_resultsdir: pathlib.Path = None,
                    upper_threshold: float = 5.0,
                    include_hybrid = False,
                    add_worst_row = False,
                    reprocess = False,
                    mode: str = "png") -> None:
    """Generate table of results.

    Args:
        resultsdir: path to results directory
        savedir: path to save directory
        benchmarking_resultsdir: path to benchmarking results directory
        upper_threshold: upper threshold for heatmap
        include_hybrid: whether to include hybrid results
        add_worst_row: whether to add a row for the worst result
    """
    bencmarking_results: Dict[str, Dict[float, pd.DataFrame]] = {}
    if benchmarking_resultsdir is not None:
        for path in benchmarking_resultsdir.glob("*.csv"):
            ccr = float(path.stem.split("_")[-1])
            recipe_name = path.stem.split("_")[0]
            df = pd.read_csv(path, index_col=0)
            df.rename(columns={"scheduler": "Scheduler", "makespan_ratio": "Makespan Ratio"}, inplace=True)
            # removesuffix "Scheduler" from scheduler names
            df["Scheduler"] = df["Scheduler"].apply(lambda x: x.removesuffix("Scheduler"))
            # rename_dict = {
            #     "Cpop": "CPoP",
            #     # "FastestNode": "FastestNode",
            # }
            # df["Scheduler"] = df["Scheduler"].apply(lambda x: rename_dict.get(x, x))
            df["Base Scheduler"] = r"\textit{Benchmarking}"
            bencmarking_results.setdefault(recipe_name, {})[ccr] = df

    for recipe_path in resultsdir.glob("*"):
        recipe_name = recipe_path.stem
        for resultsdir in recipe_path.glob("ccr_*"):
            if not resultsdir.joinpath("results.csv").exists() or reprocess:
                post_compare_all.results_to_csv(resultspath=resultsdir, outputpath=resultsdir)

            if benchmarking_resultsdir is not None:
                df_results = pd.read_csv(resultsdir.joinpath("results.csv"), index_col=0)
                df_results = pd.concat([df_results, bencmarking_results[recipe_name][ccr]], ignore_index=True)
                df_results.to_csv(resultsdir.joinpath("results.csv"))

            ccr = float(resultsdir.name.split("_")[-1])
            post_compare_all.tab_results(
                resultsdir=resultsdir,
                savedir=savedir,
                upper_threshold=upper_threshold,
                include_hybrid=include_hybrid,
                add_worst_row=add_worst_row,
                title=f"{recipe_name} (CCR = {ccr})",
                savename=f"{recipe_name}_ccr_{ccr}",
                mode=mode,
            )

experiments = {
    "compare_all": {
        "run": partial(
            exp_compare_all.run,
            output_path=thisdir.joinpath("results", "compare_all"),
        ),
        "process": partial(
            post_compare_all.results_to_csv,
            resultspath=thisdir.joinpath("results", "compare_all"),
            outputpath=thisdir.joinpath("output", "compare_all")
        ),
        "plot": partial(
            post_compare_all.tab_results,
            savedir=thisdir.joinpath("output", "compare_all"),
            upper_threshold=5.0,
        ),
    },
    "compare_wfcommons_ccr": {
        "run": partial(
            exp_wfcommons.run_many,
            output_path=thisdir.joinpath("results", "compare_wfcommons_ccr"),
            cloud="chameleon",
            ccrs=[1/5, 1/2, 1, 2, 5],
            schedulers=[
                CpopScheduler(),
                FastestNodeScheduler(),
                HeftScheduler(),
                MaxMinScheduler(),
                MinMinScheduler(),
                WBAScheduler(),
            ]
        ),
        "plot": partial(
            tab_results_ccr,
            resultsdir=thisdir.joinpath("results", "compare_wfcommons_ccr"),
            savedir=thisdir.joinpath("output", "compare_wfcommons_ccr"),
            benchmarking_resultsdir=thisdir.joinpath("results", "benchmarking_wfcommons_ccr"),
            # upper_threshold=2.0,
        )
    },
    "benchmarking": {
        "prepare": prepare_datasets.run,
        "run": partial(
            exp_benchmarking.run,
            datadir=thisdir.joinpath("datasets", "benchmarking"),
            resultsdir=thisdir.joinpath("results", "benchmarking"),
            schedulers=[
                BILScheduler(),
                CpopScheduler(),
                DuplexScheduler(),
                ETFScheduler(),
                FastestNodeScheduler(),
                FCPScheduler(),
                FLBScheduler(),
                GDLScheduler(),
                HeftScheduler(),
                MaxMinScheduler(),
                MCTScheduler(),
                METScheduler(),
                MinMinScheduler(),
                OLBScheduler(),
                WBAScheduler(),
            ],
            trim=0,
        ),
        "plot": partial(
            post_benchmarking.run,
            resultsdir=thisdir.joinpath("results", "benchmarking"),
            outputdir=thisdir.joinpath("output", "benchmarking"),
        )
    },
    "benchmarking_wfcommons_ccr": {
        "prepare": partial(
            prepare_datasets.run_wfcommons_ccrs,
            savedir=thisdir.joinpath("datasets", "benchmarking_wfcommons_ccr"),
        ),
        "run": partial(
            exp_benchmarking.run,
            datadir=thisdir.joinpath("datasets", "benchmarking_wfcommons_ccr"),
            resultsdir=thisdir.joinpath("results", "benchmarking_wfcommons_ccr"),
            schedulers=[
                CpopScheduler(),
                FastestNodeScheduler(),
                HeftScheduler(),
                MaxMinScheduler(),
                MinMinScheduler(),
                WBAScheduler(),
            ],
            trim=0,
        ),
        "plot": chain(
            partial(
                post_benchmarking.run,
                resultsdir=thisdir.joinpath("results", "benchmarking_wfcommons_ccr"),
                outputdir=thisdir.joinpath("output", "benchmarking_wfcommons_ccr", f"ccr_{ccr}"),
                glob=f"*_{ccr}.csv",
                title=f"CCR = {ccr}",
            )
            for ccr in [
                p.stem.split("_")[-1]
                for p in thisdir.joinpath("datasets", "benchmarking_wfcommons_ccr").glob("*.json")
            ]
        ),
        "plot_by_dataset": chain(
            partial(
                post_benchmarking.run,
                resultsdir=thisdir.joinpath("results", "benchmarking_wfcommons_ccr"),
                outputdir=thisdir.joinpath("output", "benchmarking_wfcommons_ccr", f"dataset_{dataset}"),
                glob=f"{dataset}_*.csv",
                title=f"Dataset = {dataset}"
            )
            for dataset in [
                p.stem.split("_")[0]
                for p in thisdir.joinpath("datasets", "benchmarking_wfcommons_ccr").glob("*.json")
            ]
        ),
    },
    "stochastic_benchmarking": {
        "prepare": partial(
            prepare_stochastic_datasets.run,
            savedir=thisdir.joinpath("datasets", "stochastic_benchmarking"),
        ),
        "run": partial(
            exp_stochastic_benchmarking.run,
            datadir=thisdir.joinpath("datasets", "stochastic_benchmarking"),
            resultsdir=thisdir.joinpath("results", "stochastic_benchmarking"),
            trim=0,
            schedulers=[
                CpopScheduler(),
                FastestNodeScheduler(),
                HeftScheduler(),
                MaxMinScheduler(),
                MinMinScheduler(),
                WBAScheduler(),
            ]
        ),
        "plot": partial(
            post_benchmarking.run,
            resultsdir=thisdir.joinpath("results", "stochastic_benchmarking"),
            outputdir=thisdir.joinpath("output", "stochastic_benchmarking"),
        )
    },
}
        
def get_parser():
    """Get parser."""
    parser = argparse.ArgumentParser(description="Run an experiment")
    subparsers = parser.add_subparsers(dest="exprunner_experiment")

    for experiment_name, experiment in experiments.items():
        subparser = subparsers.add_parser(experiment_name)
        action_subparsers = subparser.add_subparsers(dest="exprunner_action")
        for action_name, action in experiment.items():
            # if action isn't a partial, turn it into one
            if not isinstance(action, partial):
                action = partial(action)

            action_parser = action_subparsers.add_parser(action_name)
            options = action.func.__code__.co_varnames[:action.func.__code__.co_argcount]
            defaults = {
                k: v
                for k, v in zip(options[-len(action.func.__defaults__ or []):], action.func.__defaults__ or [])
            }
            # remove partially applied arguments from options
            for option in options:
                if option in action.keywords:
                    continue
                option_type = action.func.__annotations__.get(option, str)
                option_required = option not in defaults
                action_parser.add_argument(
                    f"--{option}",
                    type=option_type,
                    required=option_required,
                    default=defaults.get(option),
                    help=f"{option} ({'required' if option_required else 'default: ' + str(defaults.get(option))})",
                )

    return parser

def main():
    """Run experiment."""
    parser = get_parser()
    args = parser.parse_args()

    if args.exprunner_experiment is None or args.exprunner_action is None:
        parser.print_help()
        return
    
    experiment = experiments[args.exprunner_experiment]
    action = experiment[args.exprunner_action]
    kwargs = {k: v for k, v in vars(args).items() if not k.startswith("exprunner_")}
    action(**kwargs)


if __name__ == "__main__":
    main()