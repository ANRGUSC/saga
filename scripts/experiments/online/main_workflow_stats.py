from itertools import product
from typing import Dict, Tuple
import pathlib
import pandas as pd
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm

from saga.schedulers.data.wfcommons import get_wfcommons_instance, recipes
from saga.utils.random_variable import RandomVariable


THISDIR = pathlib.Path(__file__).resolve().parent
CSV_PATH = THISDIR / "workflow_variances.csv"
WORKFLOWS = list(recipes.keys())
N_SAMPLES = 100


def process_instance(args: Tuple[str, int]) -> Dict:
    workflow, sample = args
    task_graph, network = get_wfcommons_instance(
        recipe_name=workflow,
        ccr=5,
        max_size_multiplier=2
    )

    task_variances = [
        data["weight_rv"].var()
        for _, data in task_graph.nodes(data=True)
        if isinstance(data["weight_rv"], RandomVariable)
    ]
    dep_variances = [
        data["weight_rv"].var()
        for _, _, data in task_graph.edges(data=True)
        if isinstance(data["weight_rv"], RandomVariable)
    ]
    node_variances = [
        data["weight_rv"].var()
        for _, data in network.nodes(data=True)
        if isinstance(data["weight_rv"], RandomVariable)
    ]
    link_variances = [
        data["weight_rv"].var()
        for _, _, data in network.edges(data=True)
        if isinstance(data["weight_rv"], RandomVariable)
    ]

    task_costs = [data["weight_estimate"] for _, data in task_graph.nodes(data=True)]
    dep_costs = [data["weight_estimate"] for _, _, data in task_graph.edges(data=True)]
    node_costs = [data["weight_estimate"] for _, data in network.nodes(data=True)]
    link_costs = [data["weight_estimate"] for _, _, data in network.edges(data=True)]

    return {
        "workflow": workflow,
        "sample": sample,
        "task_variance_mean": sum(task_variances) / (len(task_variances) or 1),
        "task_variance_max": max(task_variances) if task_variances else 0,
        "dep_variance_mean": sum(dep_variances) / (len(dep_variances) or 1),
        "dep_variance_max": max(dep_variances) if dep_variances else 0,
        "node_variance_mean": sum(node_variances) / (len(node_variances) or 1),
        "node_variance_max": max(node_variances) if node_variances else 0,
        "link_variance_mean": sum(link_variances) / (len(link_variances) or 1),
        "link_variance_max": max(link_variances) if link_variances else 0,
        "task_cost_mean": sum(task_costs) / (len(task_costs) or 1),
        "task_cost_max": max(task_costs) if task_costs else 0,
        "dep_cost_mean": sum(dep_costs) / (len(dep_costs) or 1),
        "dep_cost_max": max(dep_costs) if dep_costs else 0,
        "node_cost_mean": sum(node_costs) / (len(node_costs) or 1),
        "node_cost_max": max(node_costs) if node_costs else 0,
        "link_cost_mean": sum(link_costs) / (len(link_costs) or 1),
        "link_cost_max": max(link_costs) if link_costs else 0
    }


def main():
    instances = list(product(WORKFLOWS, range(N_SAMPLES)))
    processes = 1 # max(1, int(0.9 * cpu_count()))
    print(f"Processing {len(instances)} instances across {processes} cores...")

    # Prepare CSV file with headers
    columns = [
        "workflow", "sample",
        "task_variance_mean", "task_variance_max",
        "dep_variance_mean", "dep_variance_max",
        "node_variance_mean", "node_variance_max",
        "link_variance_mean", "link_variance_max",
        "task_cost_mean", "task_cost_max",
        "dep_cost_mean", "dep_cost_max",
        "node_cost_mean", "node_cost_max",
        "link_cost_mean", "link_cost_max"
    ]
    CSV_PATH.write_text(','.join(columns) + '\n')

    manager = Manager()
    lock = manager.Lock()

    with tqdm(total=len(instances)) as pbar:
        def write_callback(row: Dict):
            with lock:
                df = pd.DataFrame([row])
                df.to_csv(CSV_PATH, mode='a', index=False, header=False)
                pbar.update(1)

        if processes == 1:
            # Single process mode
            for args in instances:
                row = process_instance(args)
                write_callback(row)
        else:
            with Pool(processes=processes) as pool:
                for args in instances:
                    pool.apply_async(process_instance, args=(args,), callback=write_callback)
                pool.close()
                pool.join()


if __name__ == "__main__":
    main()
