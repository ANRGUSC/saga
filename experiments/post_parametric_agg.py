import argparse
import pathlib
from typing import List, Set
import pandas as pd
import re

thisdir = pathlib.Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resultsdir', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)

    args = parser.parse_args()
    resultsdir = pathlib.Path(args.resultsdir)
    savepath = pathlib.Path(args.output)

    rows: List[List] = []
    for path in resultsdir.glob("*.csv"):
        _df = pd.read_csv(path)
        rows.extend(_df.values.tolist())

    df = pd.DataFrame(rows, columns=['scheduler', 'dataset', 'instance', 'makespan', 'runtime'])
    df.to_csv(savepath)


if __name__ == '__main__':
    main()