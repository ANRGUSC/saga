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


def analyze_wfcommons():
    df = pd.read_csv(thisdir / "results_wfcommons.csv")

    df_clean = remove_outliers_iqr_grouped(
        df,
        group_cols=[
            "Recipe", "Workflow Instance", "Network Instance",
            "Num Tasks", "Num Processors", "Scheduler", "Dup Factor"
        ],
        value_col="Makespan"
    )
    rows_before, rows_after = len(df), len(df_clean)
    print(f"Removed {rows_before - rows_after} outliers ({(rows_before - rows_after) / rows_before * 100:.1f}%)\n")

    df_clean["Dup Factor"] = df_clean["Dup Factor"].astype(str)

    # 1. Makespan per scheduler, grouped by dup factor, faceted by recipe (independent y-axes)
    fig = px.box(
        df_clean,
        x="Scheduler", y="Makespan", color="Dup Factor",
        facet_col="Recipe",
        title="Makespan by Scheduler and Duplication Factor (WfCommons)",
        labels={"Makespan": "Makespan"},
        template="plotly_white"
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.write_image(thisdir / "wfcommons_makespan_by_scheduler.png")
    print("Saved: wfcommons_makespan_by_scheduler.png")

    # 2. Makespan per recipe, grouped by dup factor, faceted by scheduler (independent y-axes)
    fig = px.box(
        df_clean,
        x="Recipe", y="Makespan", color="Dup Factor",
        facet_col="Scheduler",
        title="Makespan by Recipe and Duplication Factor (WfCommons)",
        labels={"Makespan": "Makespan"},
        template="plotly_white"
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.write_image(thisdir / "wfcommons_makespan_by_recipe.png")
    print("Saved: wfcommons_makespan_by_recipe.png")

    # 3. Number of processors effect, faceted by scheduler (independent y-axes per recipe via color)
    fig = px.box(
        df_clean,
        x="Num Processors", y="Makespan", color="Dup Factor",
        facet_col="Scheduler", facet_row="Recipe",
        title="Makespan vs. Number of Processors by Duplication Factor (WfCommons)",
        labels={"Makespan": "Makespan"},
        template="plotly_white"
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.write_image(thisdir / "wfcommons_num_processors_effect.png")
    print("Saved: wfcommons_num_processors_effect.png")


if __name__ == "__main__":
    analyze_wfcommons()
