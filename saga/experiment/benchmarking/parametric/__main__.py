import argparse
import random
import time
from typing import List, Tuple

import pandas as pd
import plotly.express as px
from saga.experiment.benchmarking.parametric.components import schedulers
from saga.scheduler import Scheduler
from saga.utils.random_graphs import add_random_weights, get_branching_dag, get_network
from saga.experiment import resultsdir, outputdir, datadir
import multiprocessing as mp
import networkx as nx

queue = mp.Queue()
def scale_test_scheduler(scheduler_name: str, scheduler: Scheduler, problems) -> pd.DataFrame:
    rows = []
    for network, task_graph in problems:
        t0 = time.time()
        schedule = scheduler.schedule(network, task_graph)
        dt = time.time() - t0
        makespan = max(task.end for tasks in schedule.values() for task in tasks)
        rows.append([scheduler_name, len(task_graph), makespan, dt])
        queue.put(scheduler_name)
    return pd.DataFrame(rows, columns=['scheduler', 'num_tasks', 'makespan', 'time'])


def scaling_test():    
    NUM_NODES = 10
    NUM_LEVELS = [3, 7, 10]
    NUM_SAMPLES = 10
    problems = []
    for levels in NUM_LEVELS:
        for _ in range(NUM_SAMPLES):
            task_graph = add_random_weights(get_branching_dag(levels=levels, branching_factor=2))
            network = add_random_weights(get_network(num_nodes=NUM_NODES))
            problems.append((network, task_graph))

    results = []
    total = len(problems) * len(schedulers)
    pool = mp.Pool(mp.cpu_count())
    res = pool.starmap_async(
        scale_test_scheduler,
        [(name, scheduler, problems) for name, scheduler in schedulers.items()],
        callback=results.extend,
        error_callback=lambda e: print(e)
    )
    print(f"Starting scaling test with {total} tasks")

    total = len(problems) * len(schedulers)
    count = 0
    while True:
        queue.get()
        count += 1
        print(f"Progress: {count/total*100:.2f}%" + " " * 10, end='\r')
        if count == total:
            break
    print("Progress: 100%" + " " * 10)
    
    res.wait()

    df = pd.concat(results)

    savedir = resultsdir / 'scaling_test'
    savedir.mkdir(parents=True, exist_ok=True)
    df.to_csv(savedir / 'results.csv', index=False)
    print(f"Results saved to {savedir / 'results.csv'}")
    
    df = df.groupby(['scheduler', 'num_tasks']).mean().reset_index()
    
    fig = px.scatter(
        df, x='num_tasks', y='makespan', color='scheduler',
        # log_x=True, log_y=True
        template='plotly_white',
        # no legend

    )
    # Remove legend
    fig.for_each_trace(lambda t: t.update(showlegend=False))
    fig.update_layout(
        title='Makespan vs Number of Tasks',
        xaxis_title='Number of Tasks',
        yaxis_title='Makespan',
        legend_title='Scheduler',
        # xaxis_type='log',
        # yaxis_type='log',
        showlegend=False
    )


    savedir = outputdir / 'scaling_test'
    savedir.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(savedir / 'makespan_vs_num_tasks.html'))
    print(f"Plots saved to {savedir / 'makespan_vs_num_tasks.html'}")
    fig.write_image(str(savedir / 'makespan_vs_num_tasks.png'))
    print(f"Plots saved to {savedir / 'makespan_vs_num_tasks.png'}")
        
    fig = px.line(
        df, x='num_tasks', y='time', color='scheduler',
        # log_x=True, log_y=True
        template='plotly_white',
    )
    # Remove legend
    fig.for_each_trace(lambda t: t.update(showlegend=False))
    fig.update_layout(
        title='Time vs Number of Tasks',
        xaxis_title='Number of Tasks',
        yaxis_title='Time (s)',
        legend_title='Scheduler',
        # xaxis_type='log',
        # yaxis_type='log',
        showlegend=False
    )
    fig.write_html(str(savedir / 'time_vs_num_tasks.html'))
    print(f"Plots saved to {savedir / 'time_vs_num_tasks.html'}")
    fig.write_image(str(savedir / 'time_vs_num_tasks.png'))
    print(f"Plots saved to {savedir / 'time_vs_num_tasks.png'}")

def main():
    parser = argparse.ArgumentParser(description='Benchmarking for SAGA')
    subparsers = parser.add_subparsers(help='sub-command help')

    # Scaling test
    parser_scaling_test = subparsers.add_parser('scaling_test', help='Scaling test')
    parser_scaling_test.set_defaults(func=scaling_test)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()