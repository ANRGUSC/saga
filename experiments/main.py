import argparse
import pathlib
from functools import partial

from saga.schedulers import (
    BILScheduler, CpopScheduler, DuplexScheduler,
    ETFScheduler, FastestNodeScheduler, FCPScheduler,
    FLBScheduler, GDLScheduler, HeftScheduler,
    MaxMinScheduler, MCTScheduler, METScheduler,
    MinMinScheduler, OLBScheduler, WBAScheduler
)

import exp_compare_all
import exp_wfcommons
import exp_benchmarking
import prepare_datasets
import post_benchmarking
import post_compare_all

import pathlib

thisdir = pathlib.Path(__file__).parent.absolute()

def chain(funcs):
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
                    reprocess = False) -> None:
    for recipe_path in resultsdir.glob("*"):
        for resultsdir in recipe_path.glob("ccr_*"):
            if not resultsdir.joinpath("results.csv").exists() or reprocess:
                post_compare_all.results_to_csv(resultspath=resultsdir, outputpath=resultsdir)
                
            ccr = float(resultsdir.name.split("_")[-1])
            post_compare_all.tab_results(
                resultsdir=resultsdir,
                savedir=savedir.joinpath(recipe_path.stem, f"ccr_{ccr}"),
                upper_threshold=upper_threshold,
                include_hybrid=include_hybrid,
                add_worst_row=add_worst_row
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
            upper_threshold=2.0,
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
}
        
def get_parser():
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