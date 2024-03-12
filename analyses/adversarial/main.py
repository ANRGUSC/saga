import argparse
import os
import pathlib
import importlib

import exp_compare_all
import exp_wfcommons

from post_load import results_to_csv
from post_tab_results import tab_results

def get_parser():
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers()

    parser_experiment = subparsers.add_parser("experiment", help="Run an experiment")
    parser_experiment.add_argument("experiment", choices=experiments.keys(), help="The experiment to run")
    parser_experiment.set_defaults(func='experiment')

    parser_post_process = subparsers.add_parser("process", help="Post-process results")
    parser_post_process.add_argument("experiment", choices=experiments.keys(), help="The experiment to post-process")
    parser_post_process.add_argument("--reload", action="store_true", help="Reload results from scratch")
    parser_post_process.set_defaults(func='process')

    return parser

experiments = {
    "compare_all": exp_compare_all,
    "wfcommons": exp_wfcommons
}

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.func == "experiment":
        resultspath = pathlib.Path(f"results/{args.experiment}")
        experiments[args.experiment].run(resultspath)
    elif args.func == "process":
        resultspath = pathlib.Path(f"results/{args.experiment}")
        if not resultspath.exists():
            print(f"Results for experiment {args.experiment} do not exist. Run experiment first.")
            return
        savedir = pathlib.Path(f"output/{args.experiment}")
        if args.reload or not savedir.exists():
            savedir.mkdir(exist_ok=True, parents=True)
            results_to_csv(resultspath, savedir)
        tab_results(savedir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()