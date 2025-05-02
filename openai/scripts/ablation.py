import sys
import argparse
from query_api import run_openai
import matplotlib.pyplot as plt
import pathlib
import logging
import pandas as pd
from typing import Tuple
import numpy as np

# add src folder to data path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from data import SCHEDULER_MAP, SCHEDULER_NAME_MAP

# store csv file
def store_data(df: pd.DataFrame, algorithm_1 : int, algorithm_2: int) -> None:
    thisdir = pathlib.Path(__file__).resolve().parent.parent
    savepath = thisdir / 'results' / 'ablation'

    df.to_csv(savepath / f"{SCHEDULER_NAME_MAP[algorithm_1]}_{SCHEDULER_NAME_MAP[algorithm_2]}.csv", index=False)

# plot the graphs for visualization
def plot_graphs(avg_difference: Tuple[float], std_dev: Tuple[float], algorithm_1: int, algorithm_2: int):
    
    # bar plot
    labels = [0,1,2,3,4]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, avg_difference, yerr=std_dev, capsize=8)
    plt.xlabel("Prompt Level")
    plt.ylabel("Makespan Difference Ratio Average")
    plt.title(f"Performance for different prompt level between {SCHEDULER_NAME_MAP[algorithm_1]} and {SCHEDULER_NAME_MAP[algorithm_2]}")

    for i, v in enumerate(avg_difference):
        plt.text(i, v + 0.05, f"{v:.2f}" + "%", ha='center', va='bottom')

    plt.tight_layout()

    thisdir = pathlib.Path(__file__).resolve().parent.parent
    savepath = thisdir / 'results' / 'ablation'
    savepath.mkdir(exist_ok=True)

    plt.savefig(savepath / f"{SCHEDULER_NAME_MAP[algorithm_1]}_{SCHEDULER_NAME_MAP[algorithm_2]}.png")

# ablation study for prompt level
def run_ablation(num_run : int, algorithm_1 : int, algorithm_2: int) -> None:

    # store avg makespan difference for each prompt level
    avg_difference = []
    std_dev = []

    # dataframe
    df = pd.DataFrame(columns=['Prompt_level', 'Makespan_1', 'Makespan_2', 'Makespan_diff', 'Makespan_diff_ratio'])

    for prompt_level in range(5):

        valid_example_num = 0
        valid_example = []

        while valid_example_num != num_run:
            
            try:
                # run the experiment, we don't need visualization
                example_makespan_difference, schedule_1_makespan, schedule_2_makespan = run_openai(algorithm_1, algorithm_2, prompt_level, False)
                valid_example.append(example_makespan_difference)

                print(example_makespan_difference)

                if example_makespan_difference != -1:

                    print(f"Finished {valid_example_num} iterations for prompt level {prompt_level}")

                    # add to dataframe
                    df.loc[len(df)] = [int(prompt_level), schedule_1_makespan, schedule_2_makespan, abs(schedule_1_makespan - schedule_2_makespan), example_makespan_difference]

                    valid_example_num += 1

            except Exception as e:
                print(f"Error during experiments: {e}")
        
        # add the average for this prompt level to the list
        avg_difference.append(sum(valid_example) / num_run)
        # add std
        std_dev.append(np.std(valid_example))
    
    # store data
    store_data(df, algorithm_1, algorithm_2)

    # visualization
    plot_graphs(avg_difference, std_dev, algorithm_1, algorithm_2)

def main():
    parser = argparse.ArgumentParser(description="Run ablation experiment with given algorithms and options.")

    # required argument
    parser.add_argument("algorithm_1", type=int, help="ID of the first scheduling algorithm")
    parser.add_argument("algorithm_2", type=int, help="ID of the second scheduling algorithm")

    # arguments not required
    parser.add_argument("-n", type=int, default=10, help="Prompt level")

    args = parser.parse_args()

    # run experiment
    run_ablation(
        num_run = args.n,
        algorithm_1 = args.algorithm_1,
        algorithm_2 = args.algorithm_2,
    )

if __name__ == "__main__":
    # close the info for httpx message
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()