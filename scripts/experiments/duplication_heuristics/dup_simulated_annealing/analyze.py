import json
import pathlib
import pandas as pd
import plotly.express as px
from .simulated_annealing import DupSimulatedAnnealing
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import networkx as nx
from saga import Schedule
from itertools import product
from dash import Dash, dcc, html, Input, Output
pio.renderers.default = "browser"

def plot_makespan_ratios():
    results_dir = pathlib.Path(__file__).parent / "heuristic_results_new"
    output_dir = results_dir / "makespan_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "summary.json") as f:
        data = json.load(f)

    plot_df = pd.DataFrame(data)

    plot_df["ccr"] = plot_df["ccr"].astype(str)
    plot_df["num_processors"] = plot_df["num_processors"].astype(str)

    # plot_df["recipe"] = plot_df["recipe"].replace({
    #     "montage": "Montage",
    #     "epigenomics": "Epigenomics"
    # })

    plot_df["scheduler"] = plot_df["scheduler"].replace({
        "HEFT": "HEFT",
        "CPoP": "CPOP"
    })

    ccr_order = ["0.1", "1.0", "5.0"]
    #plot_df["duplication_factor"] = plot_df[plot_df["duplication_factor"] == 2]
    fig = px.box(
        plot_df,
        x="ccr",
        y="best_makespan_ratio",
        color="recipe",
        facet_col="num_processors",
        facet_row="scheduler",
        category_orders={
            "ccr": ccr_order,
            "num_processors": ["3", "5", "8"],
            "scheduler": ["HEFT", "CPOP"],
        },
        points=False,
        template="simple_white",
        title="Simulated Annealing Makespan Ratio by CCR and Recipe (Heuristic-Guided)",
    )

    fig.add_hline(
        y=1.0,
        row="all",
        col="all",
        line_color="red",
        line_dash="dash",
        line_width=2,
    )

    fig.for_each_annotation(
        lambda a: a.update(
            text=(
                a.text
                .replace("scheduler=", "")
                .replace("num_processors=3", "3 Processors")
                .replace("num_processors=5", "5 Processors")
                .replace("num_processors=8", "8 Processors")
            )
        )
    )
    plot_df["no_improvement"] = np.isclose(
        plot_df["best_makespan_ratio"],
        1.0,
        atol=1e-9,
        rtol=0
    )

    plot_df["improved"] = (
        plot_df["best_makespan_ratio"] < 1.0 - 1e-9
    )

    plot_df["significant_improvement"] = (
        plot_df["best_makespan_ratio"] <= 0.95
    )

    summary = (
        plot_df.groupby(
            ["recipe", "scheduler", "ccr", "num_processors"]
        )
        .agg(
            mean=("best_makespan_ratio", "mean"),
            median=("best_makespan_ratio", "median"),
            min=("best_makespan_ratio", "min"),
            count=("best_makespan_ratio", "size"),
            no_improvement=("no_improvement", "sum"),
            improved=("improved", "sum"),
            significant_improvement=("significant_improvement", "sum")
        )
        .reset_index()
    )

    summary["improvement_rate"] = (
        summary["improved"] / summary["count"] * 100
    )

    summary["significant_improvement_rate"] = (
        summary["significant_improvement"] / summary["count"] * 100
    )

    print("\n=== MAKESPAN RATIO HEURISTIC-GUIDED (NON_BF) SUMMARY ===")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    fig.for_each_xaxis(lambda axis: axis.update(title_text=""))
    fig.for_each_yaxis(lambda axis: axis.update(title_text=""))

    fig.add_annotation(
        text="CCR Values",
        x=0.5,
        y=-0.10,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
    )

    fig.add_annotation(
        text="Makespan Ratio",
        x=-0.065,
        y=0.5,
        xref="paper",
        yref="paper",
        textangle=-90,
        showarrow=False,
        font=dict(size=16),
    )

    fig.update_traces(boxmean=True)

    fig.update_layout(
        legend_title_text="Recipe",
        margin=dict(l=90, r=20, t=70, b=80),
    )

    # fig.write_image(output_dir / "sa_ccr_MR.png", width=1200, height=900)
    fig.show()

def plot_all_makespan_ratios():
    base_dir = pathlib.Path(__file__).parent.parent
    original_dir = base_dir / "original_results"
    sa_dir = base_dir / "dup_simulated_annealing"

    # output_dir = base_dir / "outputs" / "comparison"
    # output_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment results
    with open(original_dir / "summary.json") as f:
        original = pd.DataFrame(json.load(f))

    with open(sa_dir / "heuristic_free_results" / "summary.json") as f:
        heuristic_free = pd.DataFrame(json.load(f))

    with open(sa_dir / "heuristic_results" / "summary.json") as f:
        heuristic_guided = pd.DataFrame(json.load(f))

    # Standardize makespan ratio column
    original = original.rename(columns={"makespan_ratio": "ratio"})
    heuristic_free = heuristic_free.rename(columns={"best_makespan_ratio": "ratio"})
    heuristic_guided = heuristic_guided.rename(columns={"best_makespan_ratio": "ratio"})

    # Label approaches
    original["approach"] = "Original"
    heuristic_free["approach"] = "Heuristic-Free SA"
    heuristic_guided["approach"] = "Heuristic-Guided SA"

    # Combine all results (including DF=2, 3, 4 for original)
    plot_df = pd.concat(
        [original, heuristic_free, heuristic_guided],
        ignore_index=True
    )

    # Format columns
    plot_df["ccr"] = plot_df["ccr"].astype(str)
    plot_df["num_processors"] = plot_df["num_processors"].astype(str)

    plot_df["recipe"] = plot_df["recipe"].replace({
        "montage": "Montage",
        "epigenomics": "Epigenomics",
        "seismology": "Seismology"
    })

    plot_df["scheduler"] = plot_df["scheduler"].replace({
        "HEFT": "HEFT",
        "CPoP": "CPOP"
    })

    # Create a combined category for CCR and approach
    plot_df["ccr_approach"] = (
        plot_df["ccr"] + " | " + plot_df["approach"]
    )

    ccr_approach_order = [
        f"{ccr} | {approach}"
        for ccr in ["0.1", "1.0", "5.0"]
        for approach in ["Original", "Heuristic-Free SA", "Heuristic-Guided SA"]
    ]

    fig = px.box(
        plot_df,
        x="ccr_approach",
        y="ratio",
        color="recipe",
        facet_col="num_processors",
        facet_row="scheduler",
        category_orders={
            "ccr_approach": ccr_approach_order,
            "num_processors": ["3", "5", "8"],
            "recipe": ["Montage", "Epigenomics", "Seismology"],
            "scheduler": ["HEFT", "CPOP"]
        },
        points=False,
        template="simple_white",
        title="Makespan Ratio Comparison Across Duplication Approaches"
    )

    fig.update_layout(boxmode="group")

    # Show mean in each box
    fig.update_traces(boxmean=True)

    # Baseline reference
    fig.add_hline(
        y=1.0,
        row="all",
        col="all",
        line_color="red",
        line_dash="dash",
        line_width=2
    )

    # Format facet labels
    fig.for_each_annotation(
        lambda a: a.update(
            text=(
                a.text
                .replace("scheduler=", "")
                .replace("num_processors=3", "3 Processors")
                .replace("num_processors=5", "5 Processors")
                .replace("num_processors=8", "8 Processors")
            )
        )
    )

    # Remove individual axis titles
    fig.for_each_xaxis(lambda axis: axis.update(title_text=""))
    fig.for_each_yaxis(lambda axis: axis.update(title_text=""))

    # Shared axis labels
    fig.add_annotation(
        text="CCR Values",
        x=0.5,
        y=-0.10,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16)
    )

    fig.add_annotation(
        text="Makespan Ratio",
        x=-0.065,
        y=0.5,
        xref="paper",
        yref="paper",
        textangle=-90,
        showarrow=False,
        font=dict(size=16)
    )

    fig.update_layout(
        legend_title_text="Recipes",
        margin=dict(l=90, r=20, t=70, b=80),
        width=1800,
        height=1200
    )

    # Save plot
    #fig.write_html(output_dir / "all_makespan_ratios.html")

    # Display plot
    fig.show()

def load_species_data(results_dir: pathlib.Path) -> pd.DataFrame:
    """Load task-level species features from completed guided SA runs."""

    records = []
    runs_dir = results_dir / "runs"
    summary_path = results_dir / "summary.json"

    summaries = {
        entry["name"]: entry
        for entry in json.loads(summary_path.read_text())
    }

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        if not (run_dir / "completed.json").exists():
            continue

        species_path = run_dir / "species_analysis.json"

        if not species_path.exists():
            continue

        recipe, scheduler, ccr, processors, trial = (
            run_dir.name.rsplit("_", 4)
        )

        if recipe not in {"montage", "epigenomics"}:
            continue

        run_summary = summaries.get(run_dir.name)

        if run_summary is None:
            continue

        best_ratio = run_summary["best_makespan_ratio"]

        species_data = json.loads(
            species_path.read_text()
        )["species"]

        for species, tasks in species_data.items():
            for task in tasks:
                records.append({
                    "run": run_dir.name,
                    "recipe": recipe,
                    "scheduler": scheduler,
                    "ccr": float(ccr.removeprefix("ccr")),
                    "processors": int(processors.removeprefix("p")),
                    "trial": int(trial.removeprefix("t")),
                    "species": species,
                    **task,
                    "best_makespan_ratio": best_ratio
                })

    if not records:
        raise ValueError(
            "No species_analysis.json data found."
        )

    df = pd.DataFrame(records)

    # Validate rather than silently discard duplicate records.
    duplicates = df.duplicated(
        subset=["run", "task_name"]
    )

    if duplicates.any():
        raise ValueError(
            f"Found {duplicates.sum()} duplicate task records. "
            "Check the species analysis data."
        )

    return df

def analyze_duplicated_species(df: pd.DataFrame) -> None:
    """Compare task duplication by species across recipes."""

    summary = (
        df.groupby(["recipe", "species"])
        .agg(
            total_task_observations=("task_name", "size"),
            duplicated_observations=("duplicated", "sum"),
            avg_task_cost=("task_cost", "mean"),
            avg_children=("num_children", "mean"),
            avg_outgoing_data=("outgoing_data_total", "mean")
        )
        .reset_index()
    )

    summary["duplication_rate"] = (
        summary["duplicated_observations"]
        / summary["total_task_observations"]
    )

    recipes = ["montage", "epigenomics"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Montage: Number of Duplicated Tasks",
            "Epigenomics: Number of Duplicated Tasks",
            "Montage: Duplication Rate",
            "Epigenomics: Duplication Rate"
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12
    )

    for col, recipe in enumerate(recipes, start=1):
        data = (
            summary[summary["recipe"] == recipe]
            .sort_values(
                "duplicated_observations",
                ascending=False
            )
        )

        print(f"\n=== {recipe.upper()}: DUPLICATED TASK SPECIES ===")
        print(
            data.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}"
            )
        )

        if data.empty:
            continue

        fig.add_trace(
            go.Bar(
                x=data["species"],
                y=data["duplicated_observations"],
                name="Duplicated Tasks",
                text=data["duplicated_observations"],
                textposition="outside",
                marker_color="#3B82F6",
                showlegend=False
            ),
            row=1,
            col=col
        )

        fig.add_trace(
            go.Bar(
                x=data["species"],
                y=data["duplication_rate"] * 100,
                name="Duplication Rate",
                text=[
                    f"{rate:.1f}%"
                    for rate in data["duplication_rate"] * 100
                ],
                textposition="outside",
                marker_color="#14B8A6",
                showlegend=False
            ),
            row=2,
            col=col
        )

        fig.update_yaxes(
            title_text="Task Observations",
            row=1,
            col=col
        )

        fig.update_yaxes(
            title_text="Duplication Rate (%)",
            range=[0, 105],
            row=2,
            col=col
        )

    fig.update_yaxes(matches="y", row=1, col=2)

    fig.update_layout(
        title="Task Duplication by Species: Montage vs. Epigenomics",
        width=1500,
        height=950,
        margin=dict(t=110, b=100)
    )

    fig.show()

def compare_task_features(df: pd.DataFrame) -> None:
    """Compare duplicated and nonduplicated task features within species."""

    features = [
        "task_cost",
        "num_parents",
        "num_children",
        "incoming_data_total",
        "outgoing_data_total"
    ]

    for recipe in ["montage", "epigenomics"]:
        recipe_df = df[df["recipe"] == recipe]

        print(f"\n=== {recipe.upper()}: TASK FEATURE COMPARISON ===")

        for species, species_df in recipe_df.groupby("species"):
            duplicated = species_df[species_df["duplicated"]]
            not_duplicated = species_df[~species_df["duplicated"]]

            if duplicated.empty or not_duplicated.empty:
                continue

            comparison = pd.DataFrame({
                "Duplicated Mean": duplicated[features].mean(),
                "Not Duplicated Mean": not_duplicated[features].mean()
            })

            comparison["Difference"] = (
                comparison["Duplicated Mean"]
                - comparison["Not Duplicated Mean"]
            )

            print(f"\nSpecies: {species}")
            print(
                f"Duplicated: {len(duplicated)} | "
                f"Not duplicated: {len(not_duplicated)}"
            )
            print(comparison.round(3).to_string())

def plot_task_feature_comparison(df: pd.DataFrame) -> None:
    """Compare mean task features across all species in each recipe."""

    features = {
        "task_cost": "Task Cost",
        "num_parents": "Number of Parents",
        "num_children": "Number of Children",
        "incoming_data_total": "Incoming Data",
        "outgoing_data_total": "Outgoing Data"
    }

    for recipe in ["montage", "epigenomics"]:
        recipe_df = df[df["recipe"] == recipe].copy()

        recipe_df["status"] = recipe_df["duplicated"].map({
            False: "Not Duplicated",
            True: "Duplicated"
        })

        # Calculate mean feature values for each species and group.
        means = (
            recipe_df.groupby(["species", "status"])[list(features)]
            .mean()
            .reset_index()
        )

        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=list(features.values()),
            vertical_spacing=0.13,
            horizontal_spacing=0.12
        )

        for index, (feature, label) in enumerate(features.items()):
            row = index // 2 + 1
            col = index % 2 + 1

            for status, color in [
                ("Not Duplicated", "#636EFA"),
                ("Duplicated", "#EF553B")
            ]:
                data = means[means["status"] == status]

                fig.add_trace(
                    go.Bar(
                        x=data["species"],
                        y=data[feature],
                        name=status,
                        marker_color=color,
                        legendgroup=status,
                        showlegend=(index == 0),
                        text=data[feature].round(2),
                        textposition="outside",
                        hovertemplate=(
                            "Species: %{x}<br>"
                            f"{label}: "
                            "%{y:.3f}<extra></extra>"
                        )
                    ),
                    row=row,
                    col=col
                )

            fig.update_yaxes(
                title_text=label,
                row=row,
                col=col
            )

            fig.update_xaxes(
                tickangle=-35,
                row=row,
                col=col
            )

        fig.update_layout(
            title=f"{recipe.title()}: Mean Task Features by Species",
            template="plotly_white",
            barmode="group",
            width=1500,
            height=1200,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=130, b=100)
        )

        fig.show()

def analyze_species_makespan(df: pd.DataFrame) -> None:
    """Summarize makespan performance for plans containing each species."""

    duplicated = df[df["duplicated"]].copy()

    species_runs = (
        duplicated.groupby(
            ["recipe", "species", "run"],
            as_index=False
        )
        .agg(
            num_duplicated_tasks=("task_name", "size"),
            best_makespan_ratio=("best_makespan_ratio", "first")
        )
    )

    summary = (
        species_runs.groupby(["recipe", "species"])
        .agg(
            runs_with_duplication=("run", "nunique"),
            total_duplicated_tasks=("num_duplicated_tasks", "sum"),
            mean_best_ratio=("best_makespan_ratio", "mean"),
            median_best_ratio=("best_makespan_ratio", "median"),
            improvement_rate=(
                "best_makespan_ratio",
                lambda values: (values < 1.0).mean()
            )
        )
        .reset_index()
    )

    summary["mean_improvement_pct"] = (
        1.0 - summary["mean_best_ratio"]
    ) * 100

    for recipe in ["montage", "epigenomics"]:
        data = (
            summary[summary["recipe"] == recipe]
            .sort_values(
                "runs_with_duplication",
                ascending=False
            )
        )

        print(f"\n=== {recipe.upper()}: SPECIES AND MAKESPAN PERFORMANCE ===")

        print(
            data.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}"
            )
        )

def plot_species_makespan(df: pd.DataFrame) -> None:
    """Compare duplication frequency and makespan across recipes."""

    duplicated = df[df["duplicated"]].copy()

    species_runs = (
        duplicated.groupby(
            ["recipe", "species", "run"],
            as_index=False
        )
        .agg(
            num_duplicated_tasks=("task_name", "size"),
            best_makespan_ratio=("best_makespan_ratio", "first")
        )
    )

    summary = (
        species_runs.groupby(["recipe", "species"])
        .agg(
            runs_with_duplication=("run", "nunique"),
            total_duplicated_tasks=("num_duplicated_tasks", "sum"),
            mean_best_ratio=("best_makespan_ratio", "mean"),
            median_best_ratio=("best_makespan_ratio", "median")
        )
        .reset_index()
    )

    recipes = ["montage", "epigenomics"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Montage",
            "Epigenomics"
        ],
        horizontal_spacing=0.12
    )

    max_tasks = summary["total_duplicated_tasks"].max()

    colors = {
        "montage": "#3B82F6",
        "epigenomics": "#14B8A6"
    }

    for col, recipe in enumerate(recipes, start=1):
        data = summary[summary["recipe"] == recipe]

        if data.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=data["runs_with_duplication"],
                y=data["mean_best_ratio"],
                mode="markers+text",
                text=data["species"],
                textposition="top center",
                name=recipe.title(),
                marker=dict(
                    size=(
                        35 * (
                            data["total_duplicated_tasks"]
                            / max_tasks
                        ) ** 0.5
                    ),
                    color=colors[recipe],
                    opacity=0.8,
                    line=dict(
                        width=1,
                        color="white"
                    )
                ),
                customdata=data[
                    [
                        "total_duplicated_tasks",
                        "median_best_ratio"
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "Species: %{text}<br>"
                    "Runs with Duplication: %{x}<br>"
                    "Mean Best Ratio: %{y:.3f}<br>"
                    "Median Best Ratio: %{customdata[1]:.3f}<br>"
                    "Total Duplicated Tasks: %{customdata[0]}"
                    "<extra></extra>"
                ),
                showlegend=False
            ),
            row=1,
            col=col
        )

        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="gray",
            row=1,
            col=col
        )

        fig.update_xaxes(
            title_text="Runs with Species Duplicated",
            row=1,
            col=col
        )

        fig.update_yaxes(
            title_text="Mean Best Makespan Ratio",
            row=1,
            col=col
        )

    fig.update_xaxes(matches="x", row=1, col=2)
    fig.update_yaxes(matches="y", row=1, col=2)

    fig.update_layout(
        title=(
            "Task Duplication Frequency vs. Makespan Performance: "
            "Montage vs. Epigenomics"
        ),
        width=1500,
        height=650,
        margin=dict(t=110, b=100)
    )

    fig.show()


def find_best_trials(recipes, schedulers):
    results_dir = pathlib.Path(__file__).parent / "heuristic_results"
    with open(results_dir / "summary.json") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    best_trials = []

    for recipe, scheduler in product(recipes, schedulers):
        subset = df[(df["recipe"] == recipe) & (df["scheduler"] == scheduler)]
        if subset.empty:
            print(f"No trials found for {recipe} + {scheduler}")
            continue
        best_trial = subset.loc[subset["best_makespan_ratio"].idxmin()]
        best_trials.append(best_trial)
    # for trial in best_trials:
    #         print(f"\nScheduler: {trial['scheduler']}")
    #         print(f"Trial: {trial['name']}")
    #         print(f"Tasks: {trial['num_tasks']}")
    #         print(f"Baseline Makespan: {trial['baseline_makespan']:.4f}")
    #         print(f"Best Makespan: {trial['best_makespan']:.4f}")
    #         print(f"Makespan Ratio: {trial['best_makespan_ratio']:.4f}")
    return best_trials

def get_task_colors(dup_plan):
    """Assign a unique color to each task selected for duplication."""

    palette = [
        "#E63946",  # Red
        "#52545D",  # grayish
        "#50D1C2",  # Tealish
        "#F4A261",  # Orange
        "#9B5DE5",  # Purple
        "#E9C46A",  # Yellow
        "#F15BB5",  # Pink
        "#43AA8B",  # Green
        "#8D6E63",  # Brown
        "#00B4D8",  # Cyan
        "#F77F00",  # Dark orange
        "#6A4C93",  # Dark purple
    ]

    return {
        name: palette[i % len(palette)]
        for i, name in enumerate(sorted(dup_plan))
    }

def create_case_study_figure():
    return make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{}, {}],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None]
        ],
        subplot_titles=(
            "Task DAG",
            "Processor Network",
            "Baseline Schedule",
            "Duplicated Schedule"
        ),
        row_heights=[0.45, 0.275, 0.275],
        vertical_spacing=0.08,
        horizontal_spacing=0.10,
        shared_xaxes=False
    )

def plot_schedules(sa: DupSimulatedAnnealing, dup_plan, task_colors, fig):
    """Compare baseline and duplicated schedules using Plotly Gantt charts."""
    print("\n=== PLOTTING DEBUG ===")
    print("Duplication plan:", dup_plan)
    print("Number of tasks selected:", len(dup_plan))
    print("Scheduler:", sa.scheduler_class.__name__)
    baseline = sa.schedule_dup_plan({})
    best = sa.schedule_dup_plan(dup_plan)

    print("\n=== SCHEDULE RECONSTRUCTION ===")
    print(f"Baseline:             {baseline.makespan:.6f}")
    print(f"Best after baseline:  {best.makespan:.6f}")
    print(f"Baseline makespan: {baseline.makespan:.4f}")
    print(f"Best makespan:     {best.makespan:.4f}")

    improvement = (
        (baseline.makespan - best.makespan)
        / baseline.makespan
        * 100
    )

    print(f"Improvement:       {improvement:.2f}%")

    processors = sorted(node.name for node in sa.network.nodes)

    def add_schedule(schedule: Schedule, row):
        for processor in processors:
            tasks = sorted(
                schedule.mapping.get(processor, []),
                key=lambda task: task.start
            )

            for task in tasks:
                is_duplicated = task.name in dup_plan
                color = task_colors.get(task.name, "steelblue")
                duration = task.end - task.start

                fig.add_trace(
                    go.Bar(
                        x=[duration],
                        y=[processor],
                        base=[task.start],
                        orientation="h",
                        width=0.65,
                        marker=dict(
                            color=color,
                            line=dict(
                                color="white",
                                width=0.5
                            )
                        ),
                        customdata=[[
                            task.name,
                            task.node,
                            task.start,
                            task.end,
                            duration,
                            is_duplicated
                        ]],
                        hovertemplate=(
                            "Task: %{customdata[0]}<br>"
                            "Processor: %{customdata[1]}<br>"
                            "Start: %{customdata[2]:.2f}<br>"
                            "End: %{customdata[3]:.2f}<br>"
                            "Duration: %{customdata[4]:.2f}<br>"
                            "Selected for duplication: %{customdata[5]}"
                            "<extra></extra>"
                        ),
                        showlegend=False
                    ),
                    row=row,
                    col=1
                )

    add_schedule(baseline, row=2)
    add_schedule(best, row=3)

    fig.update_xaxes(
        range=[0, baseline.makespan * 1.05],
        title_text="Time",
        row=2,
        col=1
    )
    fig.update_xaxes(
        range=[0, best.makespan * 1.05],
        title_text="Time",
        row=3,
        col=1
    )

    for row in [2, 3]:
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=processors[::-1],
            row=row,
            col=1
        )

    fig.layout.annotations[2].text = (
        f"Baseline Schedule (Makespan: {baseline.makespan:.2f})"
    )

    fig.layout.annotations[3].text = (
        f"Duplicated Schedule (Makespan: {best.makespan:.2f})"
    )

def plot_task_and_network(sa: DupSimulatedAnnealing, dup_plan, task_colors, fig):
    """Plot the task DAG and processor network for a selected SA trial."""

    # --------------------------------------------------
    # 1. TASK DAG
    # --------------------------------------------------

    task_graph = nx.DiGraph()

    for task in sa.task_graph.tasks:
        task_graph.add_node(task.name, cost=task.cost)

    for edge in sa.task_graph.dependencies:
        task_graph.add_edge(
            edge.source,
            edge.target,
            size=edge.size
        )

    num_tasks = task_graph.number_of_nodes()
    fig.layout.annotations[0].text = f"Task DAG ({num_tasks} Tasks)"
    # Position tasks by dependency level.
    # This places parent tasks before their children.
    layers = {
        level: sorted(generation)
        for level, generation in enumerate(
            nx.topological_generations(task_graph)
        )
    }

    task_positions = nx.multipartite_layout(
        task_graph,
        subset_key=layers,
        align="horizontal",
        scale=2
    )
    task_positions = {
        node: (x, -y) 
        for node, (x, y) in task_positions.items()
    }
    task_edge_x = []
    task_edge_y = []

    for source, target in task_graph.edges:
        x0, y0 = task_positions[source]
        x1, y1 = task_positions[target]

        task_edge_x.extend([x0, x1, None])
        task_edge_y.extend([y0, y1, None])

    task_edges = go.Scatter(
        x=task_edge_x,
        y=task_edge_y,
        mode="lines",
        line=dict(width=1, color="lightgray"),
        hoverinfo="skip",
        name="Dependencies"
    )

    task_x = []
    task_y = []
    task_node_colors = []
    task_hover = []

    duplicated_tasks = set(dup_plan)

    for name in sorted(task_graph.nodes):
        x, y = task_positions[name]

        task_x.append(x)
        task_y.append(y)

        is_duplicated = name in duplicated_tasks
        task_node_colors.append(
            task_colors.get(name, "steelblue")
        )
        children = sorted(task_graph.successors(name))
        children_text = ", ".join(children) if children else "None"
        task_hover.append(
            f"Task: {name}<br>"
            f"Selected for duplication: {is_duplicated}<br>"
            f"Cost: {task_graph.nodes[name]['cost']:.2f}<br>"
            f"Children: {children_text}"
        )

    task_nodes = go.Scatter(
        x=task_x,
        y=task_y,
        mode="markers",
        customdata=sorted(task_graph.nodes),
        marker=dict(
            size=11,
            color=task_node_colors,
            line=dict(width=1, color="white")
        ),
        text=task_hover,
        hovertemplate="%{text}<extra></extra>",
        name="Tasks"
    )

    fig.add_trace(task_edges, row=1, col=1)
    fig.add_trace(task_nodes, row=1, col=1)
    fig.update_xaxes(
        visible=False, 
        scaleanchor="y",
        scaleratio=1,
        row=1,
        col=1
    )
    fig.update_yaxes(
        visible=False, 
        row=1,
        col=1
    )

    # --------------------------------------------------
    # 2. PROCESSOR NETWORK
    # --------------------------------------------------

    network_graph = nx.Graph()

    for node in sa.network.nodes:
        network_graph.add_node(
            node.name,
            speed=node.speed
        )

    for edge in sa.network.edges:
        network_graph.add_edge(
            edge.source,
            edge.target,
            speed=edge.speed
        )

    network_positions = nx.spring_layout(
        network_graph,
        seed=42
    )

    for source, target, data in network_graph.edges(data=True):
        if source == target:
            continue
        x0, y0 = network_positions[source]
        x1, y1 = network_positions[target]

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    width=2,
                    color="lightgray"
                ),
                hovertemplate="%{text}<extra></extra>",
                showlegend=False
            ),
            row=1,
            col=2
        )
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        # show comm speed between processors
        fig.add_trace(
            go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode="text",
                text=[f"{data['speed']:.2f}"],
                textposition="middle center",
                textfont=dict(size=12, color="black"),
                hovertemplate=(
                    f"{source} ↔ {target}<br>"
                    f"Bandwidth: {data['speed']:.2f}"
                    "<extra></extra>"
                ),
                showlegend=False
            ),
            row=1,
            col=2
        )

    network_x = []
    network_y = []
    network_names = []
    network_hover = []

    for name in sorted(network_graph.nodes):
        x, y = network_positions[name]

        network_x.append(x)
        network_y.append(y)
        network_names.append(name)

        network_hover.append(
            f"Processor: {name}<br>"
            f"Speed: {network_graph.nodes[name]['speed']:.2f}"
        )

    fig.add_trace(
        go.Scatter(
            x=network_x,
            y=network_y,
            mode="markers+text",
            text=network_names,
            textposition="top center",
            marker=dict(
                size=25,
                color="steelblue",
                line=dict(width=2, color="white")
            ),
            customdata=network_hover,
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2
    )

    fig.update_xaxes(
        visible=False, 
        scaleanchor="y2",
        scaleratio=1,
        row=1,
        col=2
    )
    fig.update_yaxes(
        visible=False, 
        row=1,
        col=2
    )

def plot_best_trial(trial):
    results_dir = pathlib.Path(__file__).parent / "heuristic_results"
    runs_dir = results_dir / "runs"
    run_name = trial["name"]
    recipe = trial["recipe"]
    scheduler = trial["scheduler"]
    dup_plan = trial["best_dup_plan"]

    task_colors = get_task_colors(dup_plan)
    fig = create_case_study_figure()
    sa = DupSimulatedAnnealing.load(
        name=run_name,
        data_dir=runs_dir
    )

    plot_task_and_network(sa, dup_plan, task_colors, fig)
    plot_schedules(sa, dup_plan, task_colors, fig)
    improvement = ( 1 - trial["best_makespan_ratio"] ) * 100

    fig.update_layout(
        title=(
            f"{recipe.title()} + {scheduler} Best Trial"
            f"<br><sup>Makespan Reduction: {improvement:.2f}%</sup>"
        ),
        height=1050,
        width=1800,
        showlegend=False,
        barmode="overlay",
        plot_bgcolor="white",
        hovermode="closest"
    )
    return fig


def launch_case_study_dashboard(best_trials):
    figures = {}
    task_names_by_trial = {}

    for trial in best_trials:
        key = trial["name"]
        fig = plot_best_trial(trial)

        figures[key] = fig

        task_names_by_trial[key] = {
            name
            for trace in fig.data
            if trace.type == "scatter" and trace.name == "Tasks"
            for name in trace.customdata
        }

        app = Dash(__name__)
        app.layout = html.Div(
            [
                html.H2("Workflow Scheduling Case Studies"),

                html.Label("Select Recipe + Scheduler:"),

                dcc.Dropdown(
                    id="trial-dropdown",
                    options=[
                        {
                            "label": f"{trial['recipe'].title()} + {trial['scheduler']}",
                            "value": trial["name"]
                        }
                        for trial in best_trials
                    ],
                    value=best_trials[0]["name"],
                    clearable=False,
                    style={"width": "350px"}
                ),

                html.Br(),

                dcc.Input(
                    id="task-search",
                    type="text",
                    placeholder="Search for a task...",
                    debounce=True,
                    style={
                        "width": "350px",
                        "padding": "10px",
                        "fontSize": "16px"
                    }
                ),

                html.Div(
                    id="search-status",
                    style={"marginTop": "10px"}
                ),

                dcc.Graph(
                    id="case-study-graph",
                    figure=figures[best_trials[0]["name"]],
                    style={"height": "1050px"}
                )
            ],
            style={"padding": "20px"}
        )

        @app.callback(
            Output("case-study-graph", "figure"),
            Output("search-status", "children"),
            Input("trial-dropdown", "value"),
            Input("task-search", "value")
        )
        def highlight_task(selected_trial, search_value):
            fig = go.Figure(figures[selected_trial])

            if not search_value or not search_value.strip():
                return fig, ""

            search_value = search_value.strip()

            matches = [
                name
                for name in task_names_by_trial[selected_trial]
                if name.lower() == search_value.lower()
            ]

            if not matches:
                return fig, f"Task '{search_value}' not found."

            selected_task = matches[0]

            for trace in fig.data:

                # Highlight the matching task in the DAG.
                if trace.type == "scatter" and trace.name == "Tasks":
                    colors = list(trace.marker.color)

                    for i, name in enumerate(trace.customdata):
                        if name == selected_task:
                            colors[i] = "#FFD700"

                    trace.marker.color = colors

                # Highlight matching tasks in both Gantt charts.
                elif trace.type == "bar":
                    if trace.customdata[0][0] == selected_task:
                        trace.marker.color = "#FFD700"

            return fig, f"Highlighted task: {selected_task}"
    app.run(debug=True)


def plot_best_trials(recipes, schedulers):
    best_trials = find_best_trials(recipes, schedulers)
    for trial in best_trials:
        plot_best_trial(trial)

def main():
    # best_trials = find_best_trials(
    #     recipes=["montage", "epigenomics"],
    #     schedulers=["HEFT", "CPoP"]
    # )
    # launch_case_study_dashboard(best_trials)
    plot_makespan_ratios()
    

if __name__ == "__main__":
    main()