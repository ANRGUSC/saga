import pathlib
from typing import Dict
from matplotlib import pyplot as plt
import pandas as pd
from saga import Scheduler
from saga.schedulers.cpop import CpopScheduler
from saga.schedulers.heft import HeftScheduler
from saga.schedulers.stochastic.sheft import SheftScheduler
from saga.schedulers.stochastic.determinizer import Determinizer
from saga.utils.random_graphs import get_diamond_dag, get_network, add_rv_weights
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from saga.utils.simulator import Simulator

thisdir = pathlib.Path(__file__).parent.absolute()


def main():
    # Create random problem instance
    network = get_network(num_nodes=3)
    task_graph = get_diamond_dag()
    add_rv_weights(network)
    add_rv_weights(task_graph)

    # Draw network and task graph
    ax_network = draw_network(network, use_latex=False)
    ax_network.get_figure().savefig(thisdir / 'network.png', bbox_inches='tight')
    plt.close(ax_network.get_figure())

    ax = draw_task_graph(task_graph, use_latex=False)
    ax.get_figure().savefig(thisdir / 'task_graph.png', bbox_inches='tight')
    plt.close(ax.get_figure())
    
    # Schedule with Sheft
    scheduler = SheftScheduler()
    schedule = scheduler.schedule(network, task_graph)

    # Draw Gantt chart
    ax_schedule = draw_gantt(schedule, use_latex=False, xmax=8)
    ax_schedule.get_figure().savefig(thisdir / 'gantt.png', bbox_inches='tight')
    plt.close(ax_schedule.get_figure())

    # Note, that unlike the deterministic case, the actual schedule may vary between runs
    # due to the random weights assigned to tasks and edges. The schedule returned by Sheft
    # is an estimate of the expected schedule.

    # To evaluate how well this schedule performs, we can evaluate it on a large number of
    # deterministic instances sampled from the stochastic instance.

    # Let's do this to compare different stochastic schedulers
    schedulers: Dict[str, Scheduler] = {
        "Sheft": SheftScheduler(),
        "Mean-HEFT": Determinizer(HeftScheduler(), lambda rv: rv.mean()),
        "Mean-CPoP": Determinizer(CpopScheduler(), lambda rv: rv.mean()),
        "Mean-STD-CPoP": Determinizer(CpopScheduler(), lambda rv: rv.mean() + rv.std()),
    }

    rows = []
    for name, scheduler in schedulers.items():
        simulator = Simulator(network, task_graph, schedule)
        schedules = simulator.run(num_simulations=100)
        for i, schedule in enumerate(schedules):
            makespan = max([0 if not tasks else tasks[-1].end for tasks in schedule.values()])
            rows.append([i, name, makespan])

    # Plot boxplot of makespans
    df = pd.DataFrame(rows, columns=["Instance", "Scheduler", "Makespan"])
    fig, ax = plt.subplots()
    df.boxplot(column="Makespan", by="Scheduler", ax=ax)
    ax.set_title("Makespan of different stochastic schedulers")
    ax.set_ylabel("Makespan")
    ax.set_xlabel("Scheduler")
    fig.savefig(thisdir / 'makespan.png', bbox_inches='tight')
    plt.close(fig)




if __name__ == '__main__':
    main()

