from saga.utils.tools import check_instance_simple
from openai_api import query, visualizeGraphs, schedule
import logging
import argparse
import sys
from pathlib import Path
import traceback

# add src folder to data path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data import SCHEDULER_MAP


def run_openai(algorithm_1: int, algorithm_2: int, prompt_level: int, visualize : bool) -> float:

    # get graphs from api
    [TASK_GRAPH, NETWORK_GRAPH, explanation] = query(algorithm_1, algorithm_2, prompt_level)

    # visualize graphs
    if visualize:
        visualizeGraphs(TASK_GRAPH, NETWORK_GRAPH)

    # check to see if graphs are valid
    try:
        check_instance_simple(NETWORK_GRAPH, TASK_GRAPH)
    # catch errors and exit if there's problems
    except Exception as e:
        print(f"Error during instance check: {e}")
        return -1
    else:
        # used for debug purposes
        # logging.basicConfig(level=logging.DEBUG)

        # make schedules
        try:
            percentage_difference = schedule(algorithm_1, algorithm_2, TASK_GRAPH, NETWORK_GRAPH, visualize)

            if visualize:
                print("Explanation from ChatGPT:", explanation)
            
            return percentage_difference
            
        except Exception as e:
            traceback.print_exc() 
            return -1
        
def main():
    parser = argparse.ArgumentParser(description="Run scheduling experiment with given algorithms and options.")

    # required argument
    parser.add_argument("algorithm_1", type=int, help="ID of the first scheduling algorithm")
    parser.add_argument("algorithm_2", type=int, help="ID of the second scheduling algorithm")

    # arguments not required
    parser.add_argument("-p", type=int, default=0, help="Prompt level")
    parser.add_argument("-v", action="store_true", help="Enable visualization")

    args = parser.parse_args()

    # run experiment
    run_openai(
        algorithm_1 = args.algorithm_1,
        algorithm_2 = args.algorithm_2,
        prompt_level = args.p,
        visualize = args.v
    )

if __name__ == "__main__":
    main()
