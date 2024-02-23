import argparse
import pathlib
from typing import List, Set
import pandas as pd
import re

thisdir = pathlib.Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resultsdir', type=str, required=True, nargs='+')
    parser.add_argument('-o', '--output', type=str, required=True)

    args = parser.parse_args()
    savepath = pathlib.Path(args.output)

    print(args.resultsdir)

    rows: List[List] = []
    for _dir in args.resultsdir:
        _dir = pathlib.Path(_dir)
        for path in _dir.glob("*.csv"):
            _df = pd.read_csv(path)
            rows.extend(_df.values.tolist())

    df = pd.DataFrame(rows, columns=['scheduler', 'dataset', 'instance', 'makespan', 'runtime'])
    df.to_csv(savepath)


if __name__ == '__main__':
    main()