from functools import partial
from itertools import combinations, product
import json
import logging
import multiprocessing
import pathlib
import random
import shutil
import subprocess
import time

import pandas as pd
from saga.data import Dataset, deserialize_graph, serialize_graph
from saga.experiment import Experiment, datadir, outputdir, resultsdir
from saga.experiment.benchmarking.parametric.analyze import PARAM_NAMES, generate_interaction_plot, generate_main_effect_plot, generate_pareto_front_plot, load_data
from saga.experiment.benchmarking.prepare import load_dataset, prepare_ccr_datasets
from saga.scheduler import Scheduler
from saga.schedulers.parametric import ParametricScheduler
import requests

import networkx as nx
from typing import Any, Callable, Dict, List,  Tuple

from saga.utils.tools import standardize_network, standardize_task_graph

from saga.experiment.benchmarking.parametric.components import schedulers



queue = multiprocessing.Queue()

DEFAULT_DATASETS = [
    f"{name}_ccr_{ccr}"
    for name in ['cycles', 'chains', 'in_trees', 'out_trees']
    for ccr in [0.2, 0.5, 1, 2, 5]
]
def evaluate_instance(scheduler: Scheduler,
                      network: nx.Graph,
                      task_graph: nx.DiGraph,
                      dataset_name: str,
                      instance_num: int,
                      savepath: pathlib.Path):
    task_graph = standardize_task_graph(task_graph)
    network = standardize_network(network)
    t0 = time.time()
    schedule = scheduler.schedule(network, task_graph)
    dt = time.time() - t0
    makespan = max(task.end for tasks in schedule.values() for task in tasks)
    # dt = 2
    # makespan = 1
    df = pd.DataFrame(
        [[scheduler.__name__, dataset_name, instance_num, makespan, dt]],
        columns=["scheduler", "dataset", "instance", "makespan", "runtime"]
    )
    savepath.parent.mkdir(exist_ok=True, parents=True)
    if savepath.exists():
        df.to_csv(savepath, mode='a', header=False, index=False)
    else:
        df.to_csv(savepath, index=False)
    if queue is not None:
        queue.put(df)

def run_batch(batch_num: int,
              datadir: pathlib.Path,
              out: pathlib.Path = pathlib.Path("results.csv")):
    batch = load_batch(datadir, batch_num)
    for i, (scheduler, (dataset_name, instance_num, network, task_graph)) in enumerate(product(schedulers.values(), batch)):
        evaluate_instance(
            scheduler=scheduler,
            network=network,
            task_graph=task_graph,
            dataset_name=dataset_name,
            instance_num=instance_num,
            savepath=out
        )

    if queue is not None:
        queue.put("DONE")

def shuffle_datasets(datadir: pathlib.Path,
                     batches: int,
                     savedir: pathlib.Path,
                     trim: int = None):
    all_datasets: Dict[str, Dataset] = {}
    for dataset_path in datadir.glob("*.json"):
        all_datasets[dataset_path.stem] = load_dataset(datadir, dataset_path.stem)

    all_datasets = [
        (name, i, network, task_graph)
        for name, dataset in all_datasets.items()
        for i, (network, task_graph) in enumerate(dataset[:trim])
    ]

    # shuffle datasets
    random.shuffle(all_datasets)

    # save shuffled datasets
    savedir.mkdir(parents=True, exist_ok=True)
    for i in range(batches):
        batch_datasets = all_datasets[i::batches]
        batch_datasets = [
            (name, i, serialize_graph(network), serialize_graph(task_graph))
            for name, i, network, task_graph in batch_datasets
        ]
        savedir.joinpath(f"batch_{i}.json").write_text(json.dumps(batch_datasets))

def load_batch(datadir: pathlib.Path, batch: int) -> List[Tuple[str, int, nx.Graph, nx.DiGraph]]:
    return [
        (name, i, deserialize_graph(network), deserialize_graph(task_graph))
        for name, i, network, task_graph in json.loads(datadir.joinpath(f"batch_{batch}.json").read_text())
    ]


class ParametricExperiment(Experiment):
    def __init__(self, trim: int = 0):
        super().__init__()
        self.datadir = datadir / "parametric"
        self.resultsdir = resultsdir / "parametric"
        self.outputdir = outputdir / "parametric"

        self.trim = trim

    def prepare(self, download_url: str = None, num_batches: int = None) -> None:
        if num_batches is None:
            num_batches = multiprocessing.cpu_count()

        if download_url is not None:
            if not self.datadir.exists():
                self.datadir.mkdir(parents=True, exist_ok=True)
                print(f"Downloading dataset to {self.datadir}")
                with requests.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    with open(self.datadir / "dataset.zip", 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                shutil.unpack_archive(self.datadir / "dataset.zip", self.datadir)
            else:
                print(f"Dataset already exists in {self.datadir}")
        else:
            prepare_ccr_datasets(savedir=self.datadir, skip_existing=True)

        shuffled_datadir = self.datadir.joinpath("shuffled")
        if (not shuffled_datadir.exists() or len(list(shuffled_datadir.glob("*.json"))) != num_batches):
            if shuffled_datadir.exists():
                shutil.rmtree(shuffled_datadir)
            shuffle_datasets(self.datadir, num_batches, shuffled_datadir, trim=self.trim)
        else:
            print(f"Using existing shuffled datasets in {shuffled_datadir}")

    def run(self, num_batches: int = None, progress_callback: Callable[[Any], None] = None):
        if num_batches is None:
            num_batches = multiprocessing.cpu_count()

        if self.resultsdir.exists():
            shutil.rmtree(self.resultsdir)

        self.resultsdir.mkdir(parents=True, exist_ok=True)
        self.datadir.mkdir(parents=True, exist_ok=True)

        shuffled_datadir = self.datadir / "shuffled"

        pool = multiprocessing.Pool(processes=num_batches)
        savedir = self.resultsdir / "batch_results"
        savedir.mkdir(parents=True, exist_ok=True)

        res = pool.starmap_async(
            run_batch,
            [
                (batch, shuffled_datadir, pathlib.Path(savedir / f"results_{batch}.csv"))
                for batch in range(num_batches)
            ]
        )

        batches_done = 0
        while True:
            thing = queue.get()
            if isinstance(thing, str) and thing == "DONE":
                batches_done += 1
                if batches_done == num_batches:
                    break
            else:
                progress_callback(thing)

        print("Done!")
        res.wait()

        # concat the results
        results = pd.concat([pd.read_csv(outpath) for outpath in savedir.glob("*.csv")])
        results.to_csv(self.resultsdir / "results.csv", index=False)

    def analyze(self,
                filetype: str = "pdf",
                showfliers: bool = False,
                do_dataset_plots: bool = True,
                do_pareto_plots: bool = True,
                do_main_effect_plots: bool = True,
                do_interaction_plots: bool = True) -> None:
        df = load_data(self.resultsdir / "results.csv")
        param_names = list(set(PARAM_NAMES) - {"k_depth"})
        df["ccr"] = df["dataset"].apply(lambda x: float(x.split('_ccr_')[1]))
        df["dataset_type"] = df["dataset"].apply(lambda x: x.split('_ccr_')[0])

        if do_pareto_plots:
            generate_pareto_front_plot(df, outputdir / "parametric", filetype=filetype)

        if do_main_effect_plots:
            for param in param_names:
                generate_main_effect_plot(
                    df, param, "makespan_ratio",
                    outputdir / f"{param}-makespan-ratio.{filetype}",
                    showfliers=showfliers
                )
                generate_main_effect_plot(
                    df, param, "runtime_ratio",
                    outputdir / f"{param}-runtime-ratio.{filetype}",
                    showfliers=showfliers
                )
        if do_interaction_plots:
            for param_1, param_2 in combinations(param_names, 2):
                generate_interaction_plot(
                    df, param_1, param_2, "makespan_ratio",
                    outputdir / "interactions" / f"{param_1}-{param_2}-makespan-ratio.{filetype}"
                )
                generate_interaction_plot(
                    df, param_1, param_2, "runtime_ratio",
                    outputdir / "interactions" / f"{param_1}-{param_2}-runtime-ratio.{filetype}"
                )
            
        # Dataset specific plots
        if do_dataset_plots:
            for dataset in df["dataset"].unique():
                savedir = outputdir / "parametric" / "dataset" / dataset
                dataset_df = df[df["dataset"] == dataset]

                if do_main_effect_plots:
                    for param in param_names:
                        generate_main_effect_plot(
                            dataset_df, param, "makespan_ratio",
                            savedir / f"{param}-makespan-ratio.{filetype}",
                            showfliers=showfliers
                        )
                        generate_main_effect_plot(
                            dataset_df, param, "runtime_ratio",
                            savedir / f"{param}-runtime-ratio.{filetype}",
                            showfliers=showfliers
                        )
                if do_interaction_plots:
                    for param_1, param_2 in combinations(param_names, 2):
                        generate_interaction_plot(
                            dataset_df, param_1, param_2, "makespan_ratio",
                            savedir / "interactions" / f"{param_1}-{param_2}-makespan-ratio.{filetype}"
                        )
                        generate_interaction_plot(
                            dataset_df, param_1, param_2, "runtime_ratio",
                            savedir / "interactions" / f"{param_1}-{param_2}-runtime-ratio.{filetype}"
                        )

class ParametricExperimentSlurm(ParametricExperiment):
    def run(self, num_batches: int = None):
        if num_batches is None:
            num_batches = multiprocessing.cpu_count()

        self.resultsdir.mkdir(parents=True, exist_ok=True)
        self.datadir.mkdir(parents=True, exist_ok=True)

        shuffled_datadir = self.datadir / "shuffled"
        if not shuffled_datadir.exists():
            shuffle_datasets(self.datadir, num_batches, shuffled_datadir, trim=self.trim)
        elif len(list(shuffled_datadir.glob("*.json"))) != num_batches:
            shutil.rmtree(shuffled_datadir)
            shuffle_datasets(self.datadir, num_batches, shuffled_datadir, trim=self.trim)
        else:
            print(f"Using existing shuffled datasets in {shuffled_datadir}")

        thisdir = pathlib.Path(__file__).parent.resolve()
        proc = subprocess.Popen(
            ["sbatch", thisdir / "exp_parametric.sl"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = proc.communicate()
        print(stdout.decode())
        print(stderr.decode())
