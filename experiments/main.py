import argparse
import pathlib
from functools import partial
import re

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

from post_load import results_to_csv
from post_tab_results import tab_results

import pathlib

thisdir = pathlib.Path(__file__).parent.absolute()

def chain(funcs):
    def chained_func(*args, **kwargs):
        for func in funcs:
            func(*args, **kwargs)
    return chained_func

experiments = {
    "compare_all": {
        "run": partial(
            exp_compare_all.run,
            output_path=thisdir.joinpath("results", "compare_all"),
        ),
        "process": partial(
            results_to_csv,
            resultspath=thisdir.joinpath("results", "compare_all"),
            outputpath=thisdir.joinpath("output", "compare_all")
        ),
        "plot": partial(
            tab_results,
            savedir=thisdir.joinpath("output", "compare_all"),
            upper_threshold=2.0,
        ),
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
                # BILScheduler(),
                CpopScheduler(),
                # DuplexScheduler(),
                # ETFScheduler(),
                FastestNodeScheduler(),
                # FCPScheduler(),
                # FLBScheduler(),
                # GDLScheduler(),
                HeftScheduler(),
                MaxMinScheduler(),
                # MCTScheduler(),
                # METScheduler(),
                MinMinScheduler(),
                # OLBScheduler(),
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
# for recipe_name in ["montage", "srasearch"]:
#     for ccr in [(1,5),(2,5),(3,5),(4,5),(1,1),(5,4),(5,3),(5,2),(5,1)]:
#         experiments[f"{recipe_name}_ccr_{ccr[0]}_{ccr[1]}"] = {
#             "run": partial(
#                 exp_wfcommons.run,
#                 cloud="chameleon",
#                 recipe_name=recipe_name,
#                 ccr=ccr[0]/ccr[1],
#             ),
#             "proccess": partial(
#                 tab_results,
#                 upper_threshold=2.0,
#             ),
#         }

        
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