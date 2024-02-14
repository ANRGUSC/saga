import argparse
import pathlib
from typing import Set
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

    all_records = []
    dataset_names: Set[str] = set()
    for i, path in enumerate(resultsdir.glob("*/*.csv"), start=1):

        # try:
        #     df = pd.read_csv(path, index_col=0)
        # except pd.errors.EmptyDataError as e:
        #     print(f"Read Error: {path}")
        #     raise e
        # record = df.to_dict(orient='records')[0]
        re_output = re.compile(r'(.*?)_ccr_(\d+(?:\.\d+)?)_(\d+)')
        try:
            dataset_name, ccr, instance_num = re_output.match(path.stem).groups()
            dataset_names.add(dataset_name)
        except AttributeError as e:
            print(f"Regex Error: {path.stem}")
            raise e
        # record['ccr'] = float(ccr)

        # assert(record['dataset'].startswith(dataset_name))
        # assert(record['instance'] == int(instance_num))
        
        # all_records.append(record)

    print(f"Found {i} records: {dataset_names}")
    # df = pd.DataFrame(all_records)
    # df = df.sort_values(by=['dataset', 'ccr'])
    
    # df.to_csv(savepath)


if __name__ == '__main__':
    main()