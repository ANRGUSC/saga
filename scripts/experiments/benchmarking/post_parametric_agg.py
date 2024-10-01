import pathlib
from typing import List
import pandas as pd

thisdir = pathlib.Path(__file__).resolve().parent

def main():
    resultsdir = thisdir / 'results' / 'parametric'
    savepath = thisdir / 'results' / 'parametric.csv'

    rows: List[List] = []
    for path in resultsdir.glob("*.csv"):
        _df = pd.read_csv(path)
        rows.extend(_df.values.tolist())

    df = pd.DataFrame(rows, columns=['scheduler', 'dataset', 'instance', 'makespan', 'runtime'])
    df.to_csv(savepath)


if __name__ == '__main__':
    main()