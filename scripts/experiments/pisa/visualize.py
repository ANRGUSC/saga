from typing import Dict, Hashable, List
import streamlit as st
import pathlib
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from saga.utils.draw import draw_network, draw_task_graph, draw_gantt
from saga.pisa.simulated_annealing import SimulatedAnnealing, SimulatedAnnealingIteration
from saga import ScheduledTask


# Define the results directory
RESULTS_DIR = pathlib.Path("results")  # Update if necessary
ScheduleType = Dict[Hashable, List[ScheduledTask]]


st.set_page_config(layout="wide")


# Function to load a specific simulated annealing result
@st.cache_data
def load_result(base_scheduler: str, scheduler: str) -> SimulatedAnnealing:
    path = RESULTS_DIR / base_scheduler / f"{scheduler}.pkl"
    return pickle.loads(path.read_bytes())

# Function to extract iteration data
def extract_iteration_data(sa_result: SimulatedAnnealing):
    """Extracts iteration details from SimulatedAnnealing object."""
    data = []
    for it in sa_result.iterations:
        data.append({
            "Iteration": it.iteration,
            "Temperature": it.temperature,
            "Current Makespan Ratio": it.current_energy,
            "Neighbor Makespan Ratio": it.neighbor_energy,
            "Best Makespan Ratio": it.best_energy,
            "Accept Probability": it.accept_probability,
            "Accepted": it.accepted,
            "Change Type": type(it.change).__name__ if it.change else "None"
        })
    return pd.DataFrame(data)

# Function to visualize makespan ratio progression
def plot_progress(df):
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(6, 3))  # Set a reasonable size
    ax.plot(df["Iteration"], df["Current Makespan Ratio"], label="Current Makespan Ratio", linestyle="-")
    ax.plot(df["Iteration"], df["Best Makespan Ratio"], label="Best Makespan Ratio", linestyle="--")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Makespan Ratio")
    ax.set_title("Simulated Annealing Progression")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# Streamlit UI
def main():
    st.title("Simulated Annealing Visualization")

    # Load available schedulers
    base_schedulers = [p.name for p in RESULTS_DIR.iterdir() if p.is_dir()]
    
    if not base_schedulers:
        st.error("No results found in the 'results/' directory.")
        return

    selected_base = st.sidebar.selectbox("Select Base Scheduler", base_schedulers, index=base_schedulers.index("HEFT"))

    schedulers = [p.stem for p in (RESULTS_DIR / selected_base).glob("*.pkl")]
    selected_scheduler = st.sidebar.selectbox("Select Compared Scheduler", schedulers, index=schedulers.index("CPOP"))

    # Load and display results
    sa_result = load_result(selected_base, selected_scheduler)
    df_iterations = extract_iteration_data(sa_result)

    last_iteration: SimulatedAnnealingIteration = sa_result.iterations[-1]
    # normalize best schedules
    last_iteration.best_base_schedule = {node: [task for task in tasks if task.name not in ["__src__", "__dst__"]] for node, tasks in last_iteration.best_base_schedule.items()}
    last_iteration.best_schedule = {node: [task for task in tasks if task.name not in ["__src__", "__dst__"]] for node, tasks in last_iteration.best_schedule.items()}
    all_nodes = set(last_iteration.best_base_schedule.keys()).union(last_iteration.best_schedule.keys())
    for node in all_nodes:
        if node not in last_iteration.best_base_schedule:
            last_iteration.best_base_schedule[node] = []
        if node not in last_iteration.best_schedule:
            last_iteration.best_schedule[node] = []

    st.subheader(f"Worst-Case Problem Instance for {selected_scheduler} vs {selected_base}")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Task Graph")
        ax = draw_task_graph(last_iteration.best_task_graph)
        st.pyplot(ax.get_figure())

    with col2:
        st.write("### Network Graph")
        ax = draw_network(last_iteration.best_network, draw_colors=False)
        st.pyplot(ax.get_figure())

    st.subheader("Schedules for Worst-Case Problem Instance")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"### {selected_base} Schedule")
        ax = draw_gantt(last_iteration.best_base_schedule)
        st.pyplot(ax.get_figure())

    with col2:
        st.write(f"### {selected_scheduler} Schedule")
        ax = draw_gantt(last_iteration.best_schedule)
        st.pyplot(ax.get_figure())

    st.subheader(f"Simulated Annealing for {selected_scheduler} vs {selected_base}")
    # plot_progress(df_iterations)
    # put plot into a smaller centered container
    prog_width = 0.7
    margin = (1 - prog_width) / 2
    _, col_progress, _ = st.columns([margin, prog_width, margin])
    with col_progress:
        plot_progress(df_iterations)

    # Show iteration details
    st.subheader("Iteration Details")
    st.dataframe(df_iterations)

    # Allow user to inspect a specific iteration
    iteration_num = st.slider("Select Iteration", 0, len(sa_result.iterations) - 1, 0)
    iteration: SimulatedAnnealingIteration = sa_result.iterations[iteration_num]

    # normalize schedules
    iteration.current_base_schedule = {node: [task for task in tasks if task.name not in ["__src__", "__dst__"]] for node, tasks in iteration.current_base_schedule.items()}
    iteration.current_schedule = {node: [task for task in tasks if task.name not in ["__src__", "__dst__"]] for node, tasks in iteration.current_schedule.items()}
    all_nodes = set(iteration.current_base_schedule.keys()).union(iteration.current_schedule.keys())
    for node in all_nodes:
        if node not in iteration.current_base_schedule:
            iteration.current_base_schedule[node] = []
        if node not in iteration.current_schedule:
            iteration.current_schedule[node] = []

    iteration.neighbor_base_schedule = {node: [task for task in tasks if task.name not in ["__src__", "__dst__"]] for node, tasks in iteration.neighbor_base_schedule.items()}
    iteration.neighbor_schedule = {node: [task for task in tasks if task.name not in ["__src__", "__dst__"]] for node, tasks in iteration.neighbor_schedule.items()}
    all_nodes = set(iteration.neighbor_base_schedule.keys()).union(iteration.neighbor_schedule.keys())
    for node in all_nodes:
        if node not in iteration.neighbor_base_schedule:
            iteration.neighbor_base_schedule[node] = []
        if node not in iteration.neighbor_schedule:
            iteration.neighbor_schedule[node] = []

    iteration.best_base_schedule = {node: [task for task in tasks if task.name not in ["__src__", "__dst__"]] for node, tasks in iteration.best_base_schedule.items()}
    iteration.best_schedule = {node: [task for task in tasks if task.name not in ["__src__", "__dst__"]] for node, tasks in iteration.best_schedule.items()}
    all_nodes = set(iteration.best_base_schedule.keys()).union(iteration.best_schedule.keys())
    for node in all_nodes:
        if node not in iteration.best_base_schedule:
            iteration.best_base_schedule[node] = []
        if node not in iteration.best_schedule:
            iteration.best_schedule[node] = []

    st.write(f"### Iteration {iteration_num} Details")
    st.write(f"- **Temperature**: {iteration.temperature}")
    st.write(f"- **Current Makespan Ratio**: {iteration.current_energy}")
    st.write(f"- **Neighbor Makespan Ratio**: {iteration.neighbor_energy}")
    st.write(f"- **Best Makespan Ratio**: {iteration.best_energy}")
    st.write(f"- **Accept Probability**: {iteration.accept_probability}")
    st.write(f"- **Accepted**: {iteration.accepted}")
    st.write(f"- **Change Type**: {type(iteration.change).__name__ if iteration.change else 'None'}")

    # Display Task Graph and Network for current iteration
    st.subheader("Current Problem Instance")
    col1, col2 = st.columns(2)

    with col1:
        try:
            st.write("### Current Task Graph")
            # fig, ax = plt.subplots(figsize=(8, 6))  # Reduce size
            ax = draw_task_graph(iteration.current_task_graph)
            fig = ax.get_figure()
            st.pyplot(fig)  # Lower dpi
        except Exception as e:
            print(e)
            pass

    with col2:
        st.write("### Current Network Graph")
        ax = draw_network(iteration.current_network, draw_colors=False)
        st.pyplot(ax.get_figure())  # Lower dpi

    st.subheader("Schedules for Current Iteration")
    col1, col2 = st.columns(2)
    with col1: # schedule for base scheduler
        st.write(f"### {selected_base} Schedule")
        ax = draw_gantt(iteration.current_base_schedule)
        st.pyplot(ax.get_figure())

    with col2: # schedule for compared scheduler
        st.write(f"### {selected_scheduler} Schedule")
        ax = draw_gantt(iteration.current_schedule)
        st.pyplot(ax.get_figure())

    # Neighbour 
    st.subheader("Neighbour Problem Instance")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Task Graph")
        ax = draw_task_graph(iteration.neighbor_task_graph)
        st.pyplot(ax.get_figure())

    with col2:
        st.write("### Network Graph")
        ax = draw_network(iteration.neighbor_network, draw_colors=False)
        st.pyplot(ax.get_figure())

    st.subheader("Schedules for Neighbour Problem Instance")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"### {selected_base} Schedule")
        ax = draw_gantt(iteration.neighbor_base_schedule)
        st.pyplot(ax.get_figure())

    with col2:
        st.write(f"### {selected_scheduler} Schedule")
        ax = draw_gantt(iteration.neighbor_schedule)
        st.pyplot(ax.get_figure())


if __name__ == "__main__":
    main()
