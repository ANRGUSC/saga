import pathlib # pylint: disable=missing-module-docstring
from typing import Dict, Hashable, List

import dill as pickle
import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib import pyplot as plt
from saga.scheduler import Task
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

from simulated_annealing import SimulatedAnnealing, SimulatedAnnealingIteration

thisdir = pathlib.Path(__file__).parent.resolve()


def load_results(path: pathlib.Path) -> SimulatedAnnealing:
    """Load results from a pickle file

    Args:
        path (pathlib.Path): Path to pickle file

    Returns:
        SimulatedAnnealing: Simulated Annealing object
    """
    return pickle.load(path.open("rb"))

st.set_page_config(layout="wide")

RESULTS_PATH = "results"
runs = list(thisdir.joinpath(RESULTS_PATH).glob("**/*.pkl"))
resultsdir = thisdir.joinpath(RESULTS_PATH)
base_schedulers = [path.name for path in resultsdir.glob("*")]

def instance_view(network: nx.Graph,
                  task_graph: nx.DiGraph,
                  schedule: Dict[Hashable, List[Task]],
                  base_schedule: Dict[Hashable, List[Task]],
                  scheduler_name: str = "Scheduler",
                  base_scheduler_name: str = "Base Scheduler") -> None:
    col_task_graph, col_network = st.columns([1, 1])
    with col_task_graph:
        st.subheader("Task Graph")
        fig_task_graph, ax_task_graph = plt.subplots()
        draw_task_graph(task_graph, axis=ax_task_graph)
        fig_task_graph.set_figheight(3)
        st.pyplot(fig_task_graph, use_container_width=True)

    with col_network:
        st.subheader("Network")
        fig_network, ax_network = plt.subplots()
        draw_network(network, axis=ax_network, draw_colors=False)
        fig_network.set_figheight(3)
        st.pyplot(fig_network, use_container_width=True)

    col_schedule, col_base_schedule = st.columns([1, 1])
    with col_schedule:
        st.subheader(f"{scheduler_name} Schedule")
        fig = draw_gantt(schedule)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_base_schedule:
        st.subheader(f"{base_scheduler_name} Schedule")
        fig = draw_gantt(base_schedule)
        # set height to 200px
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def iteration_viewer(sa_run: SimulatedAnnealing) -> None:
    """Show a slider to select an iteration and show details about it

    Args:
        sa_run (SimulatedAnnealing): [description]
    """
    with st.expander("View all Iterations"):
        st.header("Iteration Viewer")
        st.subheader("Select an iteration to view details")
        iteration_index = st.slider(
            "Iteration",
            min_value=0,
            max_value=len(sa_run.iterations) - 1,
            step=1
        )

        iteration: SimulatedAnnealingIteration = sa_run.iterations[iteration_index]
        st.subheader(f"Iteration {iteration.iteration}")
        st.write(f"Change: {iteration.change}")

        df_iteration = pd.DataFrame(
            [
                [iteration.temperature, iteration.current_energy, iteration.neighbor_energy,
                iteration.best_energy, iteration.accept_probability, iteration.accepted]
            ],
            columns=[
                "Temperature", "Current Energy", "Neighbor Energy", "Best Energy",
                "Accept Probability", "Accepted"
            ],
        )

        st.table(df_iteration)
        instance_view(
            iteration.neighbor_network,
            iteration.neighbor_task_graph,
            iteration.neighbor_schedule,
            iteration.neighbor_base_schedule,
            scheduler_name=sa_run.scheduler.__class__.__name__,
            base_scheduler_name=sa_run.base_scheduler.__class__.__name__
        )

def foo(*args, **kwargs):
    print(args, kwargs)

def main():
    """Main function to show results of simulated annealing"""
    st.title("Adversarial Instance Finder")
    col_selectors, col_results = st.columns([1, 3])

    with col_selectors:
        st.subheader("Select two schedulers to compare")

        base_scheduler_name = st.selectbox("Select a base scheduler", base_schedulers)
        scheduler_choices = []
        if base_scheduler_name:
            scheduler_choices = [path.stem for path in resultsdir.glob(f"{base_scheduler_name}/*")]
        scheduler_choice = st.session_state.get('scheduler_choice', None)
        scheduler_name = st.selectbox(
            "Select a scheduler",
            scheduler_choices,
            index=(
                scheduler_choices.index(scheduler_choice)
                if scheduler_choice in scheduler_choices
                else 0
            )
        )
        st.session_state['scheduler_choice'] = scheduler_name

        st.caption(f"Simulated Annealing tries to find a worst case-scenario for {scheduler_name} against {base_scheduler_name}")

    run = resultsdir.joinpath(base_scheduler_name, scheduler_name).with_suffix(".pkl")
    with col_results:
        if run.exists():
            sa_run = load_results(run)
            records = [
                {
                    "Iteration": iteration.iteration,
                    "Temperature": iteration.temperature,
                    "Current Energy": iteration.current_energy,
                    "Neighbor Energy": iteration.neighbor_energy,
                    "Best Energy": iteration.best_energy,
                    "Accepted": iteration.accepted
                }
                for iteration in sa_run.iterations
            ]
            df_energy = pd.DataFrame.from_records(records)
            fig = px.line(df_energy, x='Iteration', y='Current Energy', title='Makespan Ratio over Iterations')
            # update y axis label
            fig.update_yaxes(title_text='Makespan Ratio')
            st.plotly_chart(fig, use_container_width=True)

            # Show Iteration Viewer
            iteration_viewer(sa_run)

            # Show Best Result
            last_iteration = sa_run.iterations[-1]
            st.header(f"Worst Makespan Ratio: {last_iteration.best_energy:.2f}")
            instance_view(
                last_iteration.best_network,
                last_iteration.best_task_graph,
                last_iteration.best_schedule,
                last_iteration.best_base_schedule,
                scheduler_name=sa_run.scheduler.__class__.__name__,
                base_scheduler_name=sa_run.base_scheduler.__class__.__name__
            )

if __name__ == '__main__':
    main()
