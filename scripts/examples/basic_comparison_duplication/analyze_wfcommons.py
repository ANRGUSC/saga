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

    df_clean["Dup Factor"] = pd.to_numeric(df_clean["Dup Factor"], errors="coerce")
    df_clean = df_clean.dropna(subset=["Dup Factor"])

    # 1. Overall percent change in makespan vs. duplication factor.
    # Use Dup Factor = 1 when available, otherwise the minimum observed factor.
    dup_agg = (
        df_clean.groupby("Dup Factor", as_index=False)["Makespan"]
        .mean()
        .sort_values("Dup Factor")
    )
    baseline_dup_factor = 1.0 if (dup_agg["Dup Factor"] == 1.0).any() else float(dup_agg["Dup Factor"].min())
    baseline_makespan = float(dup_agg.loc[dup_agg["Dup Factor"] == baseline_dup_factor, "Makespan"].iloc[0])
    dup_agg["Makespan % Change"] = (dup_agg["Makespan"] - baseline_makespan) / baseline_makespan * 100.0

    fig = px.line(
        dup_agg,
        x="Dup Factor",
        y="Makespan % Change",
        markers=True,
        title=(
            "Overall % Change in Makespan vs. Duplication Factor (WfCommons)"
            f"<br><sup>Baseline Dup Factor = {baseline_dup_factor:g}</sup>"
        ),
        labels={"Makespan % Change": "Makespan Change (%)"},
        template="plotly_white"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.write_image(thisdir / "wfcommons_overall_makespan_pct_change.png")
    print("Saved: wfcommons_overall_makespan_pct_change.png")

    df_box = df_clean.copy()
    df_box["Dup Factor"] = df_box["Dup Factor"].astype(str)

    # 2. Makespan per scheduler, grouped by dup factor, faceted by recipe (independent y-axes)
    fig = px.box(
        df_box,
        x="Scheduler", y="Makespan", color="Dup Factor",
        facet_col="Recipe",
        title="Makespan by Scheduler and Duplication Factor (WfCommons)",
        labels={"Makespan": "Makespan"},
        template="plotly_white"
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.write_image(thisdir / "wfcommons_makespan_by_scheduler.png")
    print("Saved: wfcommons_makespan_by_scheduler.png")

    # 3. Makespan per recipe, grouped by dup factor, faceted by scheduler (independent y-axes)
    fig = px.box(
        df_box,
        x="Recipe", y="Makespan", color="Dup Factor",
        facet_col="Scheduler",
        title="Makespan by Recipe and Duplication Factor (WfCommons)",
        labels={"Makespan": "Makespan"},
        template="plotly_white"
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.write_image(thisdir / "wfcommons_makespan_by_recipe.png")
    print("Saved: wfcommons_makespan_by_recipe.png")

    # 4. Number of processors effect, faceted by scheduler (independent y-axes per recipe via color)
    fig = px.box(
        df_box,
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
