import pandas as pd
import pathlib
import plotly.express as px

thisdir = pathlib.Path(__file__).parent.absolute()


def remove_outliers_iqr_grouped(df: pd.DataFrame, group_cols, value_col: str, k: float = 1.5) -> pd.DataFrame:
    df2 = df.copy()
    mask = pd.Series(True, index=df2.index)
    for name, group in df2.groupby(group_cols):
        q1 = group[value_col].quantile(0.25)
        q3 = group[value_col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        mask.loc[group.index] = group[value_col].between(lower, upper)
    return df2[mask]


def analyze_random():
    df = pd.read_csv(thisdir / "results.csv")

    df_clean = remove_outliers_iqr_grouped(
        df,
        group_cols=["CCR", "Num Nodes", "Levels", "Branching Factor", "Scheduler", "Dup Factor"],
        value_col="Makespan"
    )
    rows_before, rows_after = len(df), len(df_clean)
    print(f"Removed {rows_before - rows_after} outliers ({(rows_before - rows_after) / rows_before * 100:.1f}%)\n")

    df_clean["Dup Factor"] = df_clean["Dup Factor"].astype(str)
    df_clean["CCR"] = df_clean["CCR"].astype(str)

    # 1. Overall: makespan per scheduler, grouped by dup factor
    fig = px.box(
        df_clean,
        x="Scheduler", y="Makespan", color="Dup Factor",
        title="Makespan by Scheduler and Duplication Factor (Random Instances)",
        labels={"Makespan": "Makespan"},
        template="plotly_white"
    )
    fig.write_image(thisdir / "random_makespan_by_scheduler.png")
    print("Saved: random_makespan_by_scheduler.png")

    # 2. CCR effect: makespan per CCR value, grouped by dup factor
    fig = px.box(
        df_clean,
        x="CCR", y="Makespan", color="Dup Factor",
        facet_col="Scheduler",
        title="Makespan vs. CCR by Duplication Factor (Random Instances)",
        template="plotly_white"
    )
    fig.write_image(thisdir / "random_ccr_effect.png")
    print("Saved: random_ccr_effect.png")

    # 3. Number of processors effect
    fig = px.box(
        df_clean,
        x="Num Nodes", y="Makespan", color="Dup Factor",
        facet_col="Scheduler",
        title="Makespan vs. Number of Processors by Duplication Factor (Random Instances)",
        template="plotly_white"
    )
    fig.write_image(thisdir / "random_num_nodes_effect.png")
    print("Saved: random_num_nodes_effect.png")

    # 4. Tree depth (levels) effect
    fig = px.box(
        df_clean,
        x="Levels", y="Makespan", color="Dup Factor",
        facet_col="Scheduler",
        title="Makespan vs. Tree Depth by Duplication Factor (Random Instances)",
        template="plotly_white"
    )
    fig.write_image(thisdir / "random_levels_effect.png")
    print("Saved: random_levels_effect.png")

    # 5. Branching factor effect
    fig = px.box(
        df_clean,
        x="Branching Factor", y="Makespan", color="Dup Factor",
        facet_col="Scheduler",
        title="Makespan vs. Branching Factor by Duplication Factor (Random Instances)",
        template="plotly_white"
    )
    fig.write_image(thisdir / "random_branching_effect.png")
    print("Saved: random_branching_effect.png")


if __name__ == "__main__":
    analyze_random()
