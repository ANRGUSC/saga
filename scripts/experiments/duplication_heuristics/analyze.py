import pandas as pd
from pathlib import Path
import plotly.express as px
import numpy as np

thisdir = Path(__file__).parent.absolute()
savedir = thisdir / "outputs" 

# ANALYZE CSV FILES
def draw_random_graph_boxplots(df: pd.DataFrame, output_dir: Path):
    """
    Generate box plots for the main random graph experiment parameters.

    Creates makespan-ratio comparisons for branching factor,
    number of levels, and number of processors.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()

    plot_df["scheduler_name"] = plot_df["scheduler_name"].replace({
        "heft": "HEFT",
        "cpop": "CPOP"
    })

    plot_df["ccr"] = plot_df["ccr"].astype(str)
    plot_df["num_processors"] = plot_df["num_processors"].astype(str)

    plot_df["mode"] = plot_df["mode"].replace({
        "top-n": "Top-N",
        "bottom-n": "Bottom-N"
    })

    plot_configs = [
        ("ccr", "CCR Values"),
        ("branching_factor", "Branching Factor"),
        ("levels", "Number of Levels"),
        ("num_processors", "Number of Processors")
    ]

    for x_col, x_label in plot_configs:
        fig = px.box(
            plot_df,
            x=x_col,
            y="makespan_ratio",
            color="mode",
            facet_col="n",
            facet_row="scheduler_name",
            points=False,
            template="simple_white",
            title=f"Makespan Ratio of Top-N and Bottom-N Task Duplication Across {x_label} (Random DAGs)"
        )

        fig.add_hline( y=1.0, row="all", col="all", line_color="red", line_dash="dash", line_width=2 )

        fig.for_each_annotation(
            lambda a: a.update(
                text=(
                    a.text
                    .replace("scheduler_name=", "")
                    .replace("n=1", "1 Task Duplicated")
                    .replace("n=2", "2 Tasks Duplicated")
                    .replace("n=3", "3 Tasks Duplicated")
                )
            )
        )

        fig.for_each_xaxis(lambda axis: axis.update(title_text=""))
        fig.for_each_yaxis(lambda axis: axis.update(title_text=""))

        fig.add_annotation(
            text=x_label,
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

        fig.update_traces(boxmean=True)
        fig.update_layout( legend_title_text="Selection Mode", margin=dict( l=90, r=20, t=70, b=80 ) )
        fig.write_image(output_dir / f"{x_col}_MR.png", width=1200, height=900)
        #fig.show()

def analyze_random_graphs():
    """
    Load the random graph experiment results and generate summary plots.
    """
    df = pd.read_csv(savedir / "rand_graphs_data.csv")
    draw_random_graph_boxplots(df=df, output_dir=savedir / "random_graphs" / "makespan_analysis")

def analyze_wfcommons():
    """
    Load the WfCommons experiment results and generate summary plots.

    Creates one overall recipe comparison and one processor comparison
    for each workflow recipe.
    """
    df = pd.read_csv(savedir / "wfcommons_data.csv")
    output_dir = savedir / "wfcommons" / "makespan_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()

    plot_df["ccr"] = plot_df["ccr"].astype(str)

    plot_df["scheduler_name"] = plot_df["scheduler_name"].replace({
        "heft": "HEFT",
        "cpop": "CPOP"
    })

    ccr_order = ["0.1", "0.5", "1.0", "5.0", "10.0"]

    # overall comparison across workflow recipes
    fig = px.box(
        plot_df,
        x="ccr",
        y="makespan_ratio",
        color="recipe",
        facet_col="n",
        facet_row="scheduler_name",
        category_orders={
            "ccr": ccr_order,
            "n": [1, 2, 3],
        },
        points=False,
        template="simple_white",
        title="Makespan Ratio by CCR and Recipe (WfCommons)"
    )

    fig.add_hline( y=1.0, row="all", col="all", line_color="red", line_dash="dash", line_width=2 )

    fig.for_each_annotation(
        lambda a: a.update(
            text=(
                a.text
                .replace("scheduler_name=", "")
                .replace("n=1", "1 Task Duplicated")
                .replace("n=2", "2 Tasks Duplicated")
                .replace("n=3", "3 Tasks Duplicated")
            )
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

    fig.update_traces(boxmean=True)
    fig.update_layout(legend_title_text="Recipe", margin=dict(l=90, r=20, t=70, b=80))
    fig.write_image(output_dir / "ccr_MR.png", width=1200, height=900)

    #fig.show()

    # processor comparison for each workflow recipe
    for recipe in plot_df["recipe"].unique():
        recipe_df = plot_df[plot_df["recipe"] == recipe].copy()

        fig = px.box(
            recipe_df,
            x="ccr",
            y="makespan_ratio",
            color="scheduler_name",
            facet_col="n",
            facet_row="num_processors",
            category_orders={
                "ccr": ccr_order,
                "n": [1, 2, 3],
                "num_processors": [4, 8, 16]
            },
            points=False,
            template="simple_white",
            title=(
                f"Makespan Ratio by CCR and Processor Count "
                f"({recipe.capitalize()})"
            )
        )

        fig.add_hline( y=1.0, row="all", col="all", line_color="red", line_dash="dash", line_width=2 )

        fig.for_each_annotation(
            lambda a: a.update(
                text=(
                    a.text
                    .replace("n=1", "1 Task Duplicated")
                    .replace("n=2", "2 Tasks Duplicated")
                    .replace("n=3", "3 Tasks Duplicated")
                    .replace("num_processors=4", "4 Processors")
                    .replace("num_processors=8", "8 Processors")
                    .replace("num_processors=16", "16 Processors")
                )
            )
        )

        fig.for_each_xaxis(lambda axis: axis.update(title_text=""))
        fig.for_each_yaxis(lambda axis: axis.update(title_text=""))

        fig.add_annotation(
            text="CCR Values",
            x=0.5,
            y=-0.08,
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

        fig.update_traces(boxmean=True)
        fig.update_layout(legend_title_text="Scheduler", margin=dict(l=90, r=20, t=70, b=80))
        fig.write_image(output_dir / f"{recipe}_proc_MR.png", width=1200, height=1100)
        # fig.show()

# helper method
def task_set(value) -> set[str]:
    if pd.isna(value) or value == "":
        return set()
    return {task.strip() for task in str(value).split(",") if task.strip()}

def build_heuristic_bruteforce_comparison(heuristic_df: pd.DataFrame, bf_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Build a shared comparison DataFrame between heuristic and brute-force results.

    Computes task-selection agreement, brute-force rank of the heuristic choice,
    and accumulated Top-N duplication overlap.
    """
    heuristic_df = heuristic_df[ heuristic_df["mode"] == "top-n" ].copy()

    # one row per experiment iteration for the heuristic choice
    heuristic_results = heuristic_df[group_cols + ["task_name", "duplicated_tasks"]].rename(columns={
        "task_name": "heuristic_task",
        "duplicated_tasks": "heuristic_duplicated_tasks"
    })

    # one row per experiment iteration for the brute-force winner
    bf_winners = bf_df[bf_df["selected_by_brute_force"]][group_cols + [ "task_name", "heuristic_rank", "duplicated_tasks" ]
    ].rename(columns={
        "task_name": "bf_task",
        "heuristic_rank": "bf_winner_heuristic_rank",
        "duplicated_tasks": "bf_duplicated_tasks"
    })

    comparison = heuristic_results.merge( bf_winners, on=group_cols, how="inner" )

    # find where the heuristic-selected task ranked under brute force
    heuristic_bf_ranks = bf_df[ group_cols + [ "task_name", "brute_force_rank" ]
    ].rename(columns={
        "task_name": "heuristic_task",
        "brute_force_rank": "heuristic_task_bf_rank"
    })

    comparison = comparison.merge( heuristic_bf_ranks, on=group_cols + ["heuristic_task"], how="left" )

    # did both methods select the same task this iteration?
    comparison["exact_match"] = ( comparison["heuristic_task"] == comparison["bf_task"] )

    # was the brute-force winner ranked within the heuristic Top-3?
    comparison["bf_winner_in_heuristic_top3"] = ( comparison["bf_winner_heuristic_rank"] <= 3 )
    
    # was the heuristic choice within the brute-force Top-3?
    comparison["heuristic_task_in_bf_top3"] = ( comparison["heuristic_task_bf_rank"] <= 3 )

    # group the brute-force rank of the heuristic choice
    comparison["bf_rank_group"] = np.select(
        [ comparison["heuristic_task_bf_rank"] == 1, comparison["heuristic_task_bf_rank"] == 2, comparison["heuristic_task_bf_rank"] == 3 ],
        [ "Rank 1", "Rank 2", "Rank 3" ],
        default="Rank 4+"
    )

    # compare the accumulated duplicated task sets after each iteration
    comparison["top_n_overlap"] = comparison.apply(
        lambda row: (len(task_set(row["heuristic_duplicated_tasks"]) & task_set(row["bf_duplicated_tasks"])) / row["n"]),
        axis=1
    )

    return comparison

def compare_random_graph_heuristic_bruteforce():
    """
    Compare heuristic task selection against brute-force validation
    for the random graph experiments.

    Generates plots for heuristic ranking quality, accumulated Top-N
    duplication overlap, and the brute-force rank of heuristic choices.
    """
    heuristic_df = pd.read_csv(savedir / "rand_graphs_data.csv")
    bf_df = pd.read_csv(savedir / "rand_graphs_brute_force_data.csv")
    output_dir = savedir / "random_graphs" / "heuristic_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    group_cols = [ "scheduler_name", "experiment_num", "dag_type", "ccr", "levels", "branching_factor", "num_processors", "n" ]
    comparison = build_heuristic_bruteforce_comparison( heuristic_df=heuristic_df, bf_df=bf_df, group_cols=group_cols )

    # plot 1: Heuristic ranking quality
    rank_metrics = (
        comparison.groupby( ["scheduler_name", "ccr", "n"] )
        .agg(
            exact_match=( "exact_match", "mean" ),
            bf_winner_top3=( "bf_winner_in_heuristic_top3", "mean" ),
            heuristic_choice_top3=( "heuristic_task_in_bf_top3", "mean" )
        )
        .reset_index()
    )

    rank_metrics = rank_metrics.melt(
        id_vars=[ "scheduler_name", "ccr", "n" ],
        value_vars=[
            "exact_match",
            "bf_winner_top3",
            "heuristic_choice_top3"
        ],
        var_name="metric",
        value_name="rate"
    )

    rank_metrics["rate"] *= 100

    rank_metrics["metric"] = rank_metrics["metric"].replace({
        "exact_match": "Iteration Top-1 Match",
        "bf_winner_top3": "BF Winner in Heuristic Top-3",
        "heuristic_choice_top3": "Heuristic Choice in BF Top-3"
    })

    rank_metrics["scheduler_name"] = (
        rank_metrics["scheduler_name"].replace({
            "heft": "HEFT",
            "cpop": "CPOP"
        })
    )

    rank_metrics["ccr"] = rank_metrics["ccr"].astype(str)

    fig = px.bar(
        rank_metrics,
        x="ccr",
        y="rate",
        color="metric",
        facet_row="scheduler_name",
        facet_col="n",
        barmode="group",
        text="rate",
        labels={ "metric": "Metric" },
        category_orders={
            "ccr": ["0.1", "1.0", "10.0"],
            "n": [1, 2, 3]
        },
        title="Heuristic Ranking Quality vs Brute Force (Random Graphs)",
        template="simple_white"
    )

    fig.for_each_annotation(
        lambda a: a.update(
            text=(
                a.text
                .replace("scheduler_name=", "")
                .replace("n=1", "1 Task Duplicated")
                .replace("n=2", "2 Tasks Duplicated")
                .replace("n=3", "3 Tasks Duplicated")
            )
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
        font=dict(size=16)
    )

    fig.add_annotation(
        text="Rate %",
        x=-0.065,
        y=0.5,
        xref="paper",
        yref="paper",
        textangle=-90,
        showarrow=False,
        font=dict(size=16)
    )

    fig.update_traces(
        texttemplate="%{text:.0f}%",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=8),
        constraintext="none"
    )

    fig.update_yaxes( range=[0, 100] )
    fig.update_layout( legend_title_text="Metric", uniformtext_minsize=7, uniformtext_mode="hide" )
    fig.write_image( output_dir / "heu_rank_comparison.png", width=1500, height=800 )
    #fig.show()

    # plot 2: Accumulated Top-N duplication overlap
    overlap_summary = (
        comparison.groupby(["scheduler_name", "ccr", "n"])["top_n_overlap"]
        .mean()
        .reset_index()
    )

    overlap_summary["top_n_overlap"] *= 100

    overlap_summary["scheduler_name"] = (
        overlap_summary["scheduler_name"].replace({
            "heft": "HEFT",
            "cpop": "CPOP"
        })
    )

    overlap_summary["ccr"] = ( overlap_summary["ccr"].astype(str) )

    fig = px.bar(
        overlap_summary,
        x="ccr",
        y="top_n_overlap",
        color="scheduler_name",
        facet_col="n",
        barmode="group",
        text="top_n_overlap",
        labels={ "scheduler_name": "Scheduler" },
        category_orders={
            "ccr": ["0.1", "1.0", "10.0"],
            "n": [1, 2, 3]
        },
        title="Top-N Duplication Overlap (Random Graphs)",
        template="simple_white"
    )
    
    fig.for_each_annotation(
        lambda a: a.update(
            text=(
                a.text
                .replace("scheduler_name=", "")
                .replace("n=1", "1 Task Duplicated")
                .replace("n=2", "2 Tasks Duplicated")
                .replace("n=3", "3 Tasks Duplicated")
            )
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
        font=dict(size=16)
    )

    fig.add_annotation(
        text="Avg Overlap %",
        x=-0.065,
        y=0.5,
        xref="paper",
        yref="paper",
        textangle=-90,
        showarrow=False,
        font=dict(size=16)
    )

    fig.update_traces(
        texttemplate="%{text:.0f}%",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=9),
        constraintext="none"
    )

    fig.update_yaxes( range=[0, 100] )
    fig.update_layout( legend_title_text="Scheduler", uniformtext_minsize=8, uniformtext_mode="hide" )
    fig.write_image( output_dir / "top_n_dup_overlap.png", width=1300, height=600 )
    #fig.show()

    # plot 3: Brute-force rank of heuristic choice
    rank_distribution = (
        comparison.groupby( [ "scheduler_name", "ccr", "n", "bf_rank_group" ] )
        .size()
        .reset_index(name="count")
    )

    rank_distribution["rate"] = (
        rank_distribution["count"] / rank_distribution.groupby( [ "scheduler_name", "ccr", "n" ])["count"].transform("sum")
        * 100
    )

    rank_distribution["scheduler_name"] = (
        rank_distribution["scheduler_name"].replace({
            "heft": "HEFT",
            "cpop": "CPOP"
        })
    )

    rank_distribution["ccr"] = ( rank_distribution["ccr"].astype(str) )

    fig = px.bar(
        rank_distribution,
        x="ccr",
        y="rate",
        color="bf_rank_group",
        facet_row="scheduler_name",
        facet_col="n",
        barmode="stack",
        text="rate",
        category_orders={
            "ccr": ["0.1", "1.0", "10.0"],
            "n": [1, 2, 3],
            "bf_rank_group": [
                "Rank 1",
                "Rank 2",
                "Rank 3",
                "Rank 4+"
            ]
        },
        labels={ "bf_rank_group": "Brute-Force Rank" },
        title="Brute-Force Rank of Heuristic Choice (Random Graphs)",
        template="simple_white"
    )
    
    fig.for_each_annotation(
        lambda a: a.update(
            text=(
                a.text
                .replace("scheduler_name=", "")
                .replace("n=1", "1 Task Duplicated")
                .replace("n=2", "2 Tasks Duplicated")
                .replace("n=3", "3 Tasks Duplicated")
            )
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
        font=dict(size=16)
    )

    fig.add_annotation(
        text="Percentage %",
        x=-0.065,
        y=0.5,
        xref="paper",
        yref="paper",
        textangle=-90,
        showarrow=False,
        font=dict(size=16)
    )

    fig.update_traces(
        texttemplate="%{text:.0f}%",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=8),
        constraintext="none"
    )

    fig.update_yaxes( range=[0, 100] )
    fig.update_layout( legend_title_text="Brute-Force Rank", uniformtext_minsize=7, uniformtext_mode="hide" )
    fig.write_image( output_dir / "heu_choice_bf_rank_distribution.png", width=1500, height=800 )
    #fig.show()

def compare_wfcommons_heuristic_bruteforce():
    """
    Compare heuristic task selection against brute-force validation
    for the WfCommons experiments.

    Generates plots for heuristic ranking quality, accumulated Top-N
    duplication overlap, and the brute-force rank of heuristic choices.
    """
    heuristic_df = pd.read_csv( savedir / "wfcommons_data.csv" )
    bf_df = pd.read_csv( savedir / "wfcommons_brute_force_data.csv" )
    output_dir = savedir / "wfcommons" / "heuristic_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    group_cols = [ "scheduler_name", "recipe", "workflow_instance", "ccr", "num_processors", "n" ]
    comparison = build_heuristic_bruteforce_comparison( heuristic_df=heuristic_df, bf_df=bf_df, group_cols=group_cols )


    # plot 1: Heuristic ranking quality
    rank_metrics = (
        comparison.groupby( [ "recipe", "scheduler_name", "n" ] )
        .agg(
            exact_match=( "exact_match", "mean" ),
            bf_winner_top3=( "bf_winner_in_heuristic_top3", "mean" ),
            heuristic_choice_top3=( "heuristic_task_in_bf_top3", "mean" )
        )
        .reset_index()
    )

    rank_metrics = rank_metrics.melt(
        id_vars=[ "recipe", "scheduler_name", "n" ],
        value_vars=[ "exact_match", "bf_winner_top3", "heuristic_choice_top3" ],
        var_name="metric",
        value_name="rate"
    )

    rank_metrics["rate"] *= 100

    rank_metrics["metric"] = rank_metrics["metric"].replace({
        "exact_match": "Iteration Top-1 Match",
        "bf_winner_top3": "BF Winner in Heuristic Top-3",
        "heuristic_choice_top3": "Heuristic Choice in BF Top-3"
    })

    rank_metrics["scheduler_name"] = (
        rank_metrics["scheduler_name"].replace({
            "heft": "HEFT",
            "cpop": "CPOP"
        })
    )

    fig = px.bar(
        rank_metrics,
        x="recipe",
        y="rate",
        color="metric",
        facet_row="scheduler_name",
        facet_col="n",
        barmode="group",
        text="rate",
        category_orders={
            "n": [1, 2, 3],
            "recipe": [
                "epigenomics",
                "montage",
                "seismology"
            ],
        },
        labels={ "metric": "Metric" },
        title="Heuristic Ranking Quality vs Brute Force (WfCommons)",
        template="simple_white"
    )
    
    fig.for_each_annotation(
        lambda a: a.update(
            text=(
                a.text
                .replace("scheduler_name=", "")
                .replace("n=1", "1 Task Duplicated")
                .replace("n=2", "2 Tasks Duplicated")
                .replace("n=3", "3 Tasks Duplicated")
            )
        )
    )

    fig.for_each_xaxis(lambda axis: axis.update(title_text=""))
    fig.for_each_yaxis(lambda axis: axis.update(title_text=""))

    fig.add_annotation(
        text="Recipe",
        x=0.5,
        y=-0.10,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16)
    )

    fig.add_annotation(
        text="Rate %",
        x=-0.065,
        y=0.5,
        xref="paper",
        yref="paper",
        textangle=-90,
        showarrow=False,
        font=dict(size=16)
    )

    fig.update_traces(
        texttemplate="%{text:.0f}%",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=8),
        constraintext="none"
    )

    fig.update_yaxes( range=[0, 100] )
    fig.update_layout( legend_title_text="Metric", uniformtext_minsize=7, uniformtext_mode="hide" )
    fig.write_image( output_dir / "heu_rank_comparison.png", width=1300, height=1000 )
    #fig.show()

    # plot 2: Accumulated Top-N duplication overlap
    overlap_summary = (
        comparison.groupby( [ "recipe", "scheduler_name", "n" ])["top_n_overlap"]
        .mean()
        .reset_index()
    )

    overlap_summary["top_n_overlap"] *= 100

    overlap_summary["scheduler_name"] = (
        overlap_summary["scheduler_name"].replace({
            "heft": "HEFT",
            "cpop": "CPOP"
        })
    )

    fig = px.bar(
        overlap_summary,
        x="recipe",
        y="top_n_overlap",
        color="scheduler_name",
        facet_col="n",
        barmode="group",
        text="top_n_overlap",
        category_orders={
            "n": [1, 2, 3],
            "recipe": [
                "epigenomics",
                "montage",
                "seismology"
            ],
        },
        labels={ "scheduler_name": "Scheduler" },
        title="Top-N Duplication Overlap (WfCommons)",
        template="simple_white"
    )

    fig.for_each_annotation(
        lambda a: a.update(
            text=(
                a.text
                .replace("scheduler_name=", "")
                .replace("n=1", "1 Task Duplicated")
                .replace("n=2", "2 Tasks Duplicated")
                .replace("n=3", "3 Tasks Duplicated")
            )
        )
    )

    fig.for_each_xaxis(lambda axis: axis.update(title_text=""))
    fig.for_each_yaxis(lambda axis: axis.update(title_text=""))

    fig.add_annotation(
        text="Recipe",
        x=0.5,
        y=-0.10,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16)
    )

    fig.add_annotation(
        text="Avg Overlap %",
        x=-0.065,
        y=0.5,
        xref="paper",
        yref="paper",
        textangle=-90,
        showarrow=False,
        font=dict(size=16)
    )

    fig.update_traces(
        texttemplate="%{text:.0f}%",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=9),
        constraintext="none"
    )

    fig.update_yaxes( range=[0, 100] )
    fig.update_layout( legend_title_text="Scheduler", uniformtext_minsize=8, uniformtext_mode="hide" )
    fig.write_image( output_dir / "top_n_dup_overlap.png", width=1100, height=900 )
    #fig.show()

    # plot 3: Brute-force rank of heuristic choice
    rank_distribution = (
        comparison.groupby( [ "recipe", "scheduler_name", "n", "bf_rank_group" ] )
        .size()
        .reset_index(name="count")
    )

    rank_distribution["rate"] = (
        rank_distribution["count"] / rank_distribution.groupby( [ "recipe", "scheduler_name", "n" ])["count"].transform("sum")
        * 100
    )

    rank_distribution["scheduler_name"] = (
        rank_distribution["scheduler_name"].replace({
            "heft": "HEFT",
            "cpop": "CPOP"
        })
    )

    fig = px.bar(
        rank_distribution,
        x="recipe",
        y="rate",
        color="bf_rank_group",
        facet_row="scheduler_name",
        facet_col="n",
        barmode="stack",
        text="rate",
        category_orders={
            "n": [1, 2, 3],
            "recipe": [
                "epigenomics",
                "montage",
                "seismology"
            ],
            "bf_rank_group": [
                "Rank 1",
                "Rank 2",
                "Rank 3",
                "Rank 4+"
            ],
        },
        labels={ "bf_rank_group": "Brute-Force Rank" },
        title="Brute-Force Rank of Heuristic Choice (WfCommons)",
        template="simple_white"
    )

    fig.for_each_annotation(
        lambda a: a.update(
            text=(
                a.text
                .replace("scheduler_name=", "")
                .replace("n=1", "1 Task Duplicated")
                .replace("n=2", "2 Tasks Duplicated")
                .replace("n=3", "3 Tasks Duplicated")
            )
        )
    )

    fig.for_each_xaxis(lambda axis: axis.update(title_text=""))
    fig.for_each_yaxis(lambda axis: axis.update(title_text=""))

    fig.add_annotation(
        text="Recipe",
        x=0.5,
        y=-0.10,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16)
    )

    fig.add_annotation(
        text="Percentage %",
        x=-0.065,
        y=0.5,
        xref="paper",
        yref="paper",
        textangle=-90,
        showarrow=False,
        font=dict(size=16)
    )

    fig.update_traces(
        texttemplate="%{text:.0f}%",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=8),
        constraintext="none"
    )

    fig.update_yaxes( range=[0, 100] )
    fig.update_layout( legend_title_text="Brute-Force Rank", uniformtext_minsize=7, uniformtext_mode="hide" )
    fig.write_image( output_dir / "heu_choice_bf_rank_distribution.png", width=1300, height=1000 )
    #fig.show()


if __name__ == "__main__":
    analyze_random_graphs()
    #analyze_wfcommons()
    #compare_random_graph_heuristic_bruteforce()
    #compare_wfcommons_heuristic_bruteforce()