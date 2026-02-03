import pandas as pd
import pathlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import glob
import os

thisdir = pathlib.Path(__file__).parent.absolute()


def remove_outliers_iqr_grouped(df: pd.DataFrame, group_cols, value_col: str, k: float = 1.5) -> pd.DataFrame:
    """
    Remove rows whose value_col is an outlier within its group defined by group_cols,
    using the IQR rule with multiplier k.
    """
    df2 = df.copy()
    mask = pd.Series(True, index=df2.index)
    for name, group in df2.groupby(group_cols):
        q1 = group[value_col].quantile(0.25)
        q3 = group[value_col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        mask.loc[group.index] = group[value_col].between(lower, upper)
    return df2[mask]


def analyze_results():
    """Load results and generate focused plots on duplication benefits."""
    
    # Clean up old plot files (HTML, PNG, PDF only - keep CSV files)
    print("Cleaning up old plots...")
    old_files = list(glob.glob(str(thisdir / "*.html"))) + \
                list(glob.glob(str(thisdir / "*.png"))) + \
                list(glob.glob(str(thisdir / "*.pdf")))
    
    for file in old_files:
        try:
            os.remove(file)
        except OSError:
            pass
    
    if old_files:
        print(f"Deleted {len(old_files)} old plot files\n")
    
    df = pd.read_csv(thisdir / "results.csv")
    
    # Remove outliers - group by ALL experimental variables to ensure we only remove
    # true outliers within each specific configuration
    df_clean = remove_outliers_iqr_grouped(
        df, 
        group_cols=["Scheduler", "Dup Factor", "CCR", "Num Nodes", "Levels", "Branching Factor"],
        value_col="Makespan"
    )
    
    rows_before = len(df)
    rows_after = len(df_clean)
    print(f"Removed {rows_before - rows_after} outliers ({((rows_before - rows_after) / rows_before * 100):.1f}%)\n")
    
    df_clean["Dup Factor"] = df_clean["Dup Factor"].astype(str)
    
    print(f"Creating plots to analyze duplication benefits...")
    
    # 1. Overall comparison: With vs Without Duplication
    fig1 = px.box(
        df_clean,
        x="Scheduler",
        y="Makespan",
        color="Dup Factor",
        template="plotly_white",
        title="Effect of Duplication on Makespan",
        labels={"Dup Factor": "Duplication Factor"},
        category_orders={"Dup Factor": ["1", "2"]}
    )
    fig1.update_layout(height=500)
    fig1.write_html(thisdir / "duplication_overall.html")
    fig1.write_image(thisdir / "duplication_overall.png", width=800, height=500)
    
    # 2. CCR effect with duplication
    fig2 = px.box(
        df_clean,
        x="CCR",
        y="Makespan",
        color="Dup Factor",
        facet_col="Scheduler",
        template="plotly_white",
        title="CCR Impact on Makespan (with/without Duplication)",
        labels={"Dup Factor": "Duplication Factor"},
        category_orders={"Dup Factor": ["1", "2"]}
    )
    fig2.update_layout(height=500)
    fig2.write_html(thisdir / "ccr_effect.html")
    fig2.write_image(thisdir / "ccr_effect.png", width=1000, height=500)
    
    # 3. Number of nodes effect with duplication
    fig3 = px.box(
        df_clean,
        x="Num Nodes",
        y="Makespan",
        color="Dup Factor",
        facet_col="Scheduler",
        template="plotly_white",
        title="Network Size Impact on Makespan (with/without Duplication)",
        labels={"Dup Factor": "Duplication Factor"},
        category_orders={"Dup Factor": ["1", "2"]}
    )
    fig3.update_layout(height=500)
    fig3.write_html(thisdir / "num_nodes_effect.html")
    fig3.write_image(thisdir / "num_nodes_effect.png", width=1000, height=500)
    
    # 4. Branching factor effect with duplication
    fig4 = px.box(
        df_clean,
        x="Branching Factor",
        y="Makespan",
        color="Dup Factor",
        facet_col="Scheduler",
        template="plotly_white",
        title="Branching Factor Impact on Makespan (with/without Duplication)",
        labels={"Dup Factor": "Duplication Factor"},
        category_orders={"Dup Factor": ["1", "2"]}
    )
    fig4.update_layout(height=500)
    fig4.write_html(thisdir / "branching_effect.html")
    fig4.write_image(thisdir / "branching_effect.png", width=1000, height=500)
    
    # 5. Levels effect with duplication
    fig5 = px.box(
        df_clean,
        x="Levels",
        y="Makespan",
        color="Dup Factor",
        facet_col="Scheduler",
        template="plotly_white",
        title="Graph Depth (Levels) Impact on Makespan (with/without Duplication)",
        labels={"Dup Factor": "Duplication Factor"},
        category_orders={"Dup Factor": ["1", "2"]}
    )
    fig5.update_layout(height=500)
    fig5.write_html(thisdir / "levels_effect.html")
    fig5.write_image(thisdir / "levels_effect.png", width=1000, height=500)
    
    # 6. Improvement summary: % reduction in makespan from duplication
    df_summary = df_clean.groupby(
        ["Scheduler", "CCR", "Num Nodes", "Levels", "Branching Factor", "Dup Factor"]
    )["Makespan"].mean().reset_index()
    
    df_wide = df_summary.pivot_table(
        index=["Scheduler", "CCR", "Num Nodes", "Levels", "Branching Factor"],
        columns="Dup Factor",
        values="Makespan"
    ).reset_index()
    
    df_wide["Improvement %"] = ((df_wide["1"] - df_wide["2"]) / df_wide["1"]) * 100
    
    fig6 = px.bar(
        df_wide.groupby("Scheduler")["Improvement %"].mean().reset_index(),
        x="Scheduler",
        y="Improvement %",
        template="plotly_white",
        title="Average Makespan Improvement from Duplication",
        text_auto='.1f',
        labels={"Improvement %": "Avg % Improvement"}
    )
    fig6.update_traces(textposition='outside')
    fig6.add_hline(y=0, line_dash="dash", line_color="gray")
    fig6.update_layout(height=400)
    fig6.write_html(thisdir / "improvement_summary.html")
    fig6.write_image(thisdir / "improvement_summary.png", width=600, height=400)
    
    # Save detailed improvement stats
    improvement_stats = df_wide.groupby("Scheduler").agg({
        "Improvement %": ["mean", "std", "min", "max"]
    }).round(2)
    improvement_stats.to_csv(thisdir / "improvement_statistics.csv")
    
    print(f"\n✓ Generated 6 plots:")
    print(f"  1. duplication_overall - Overall effect of duplication")
    print(f"  2. ccr_effect - How CCR affects duplication benefits")
    print(f"  3. num_nodes_effect - How network size affects duplication benefits")
    print(f"  4. branching_effect - How branching factor affects duplication benefits")
    print(f"  5. levels_effect - How graph depth affects duplication benefits")
    print(f"  6. improvement_summary - Average % improvement from duplication")
    print(f"\n✓ Summary statistics saved to improvement_statistics.csv")
    print(f"\nAll files saved to: {thisdir}")


if __name__ == '__main__':
    analyze_results()




