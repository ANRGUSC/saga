from itertools import combinations
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

# ---------------- Config ----------------
THISDIR = pathlib.Path(__file__).resolve().parent
CSV_PATH = THISDIR / "results-full.csv"
OUTDIR = THISDIR / "plots-full"
OUTDIR.mkdir(exist_ok=True)
# ----------------------------------------

def remove_outliers(df, columns):
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)

    print("Processing...")
    # Pivot to bring scheduler types into columns
    pivot = df.pivot_table(
        index=["workflow", "ccr", "sample", "scheduler", "estimate_method"],
        columns="scheduler_type",
        values="makespan"
    ).reset_index()

    # Drop incomplete entries
    pivot = pivot.dropna(subset=["Offline", "Online", "Naive Online"])

    # Get best makespan per workflow/ccr/sample (across all schedulers)
    df_min = df.groupby(["workflow", "ccr", "sample"])["makespan"].min().reset_index()
    df_min = df_min.rename(columns={"makespan": "best_makespan"})

    # Merge into pivoted data
    pivot = pd.merge(pivot, df_min, on=["workflow", "ccr", "sample"], how="left")

    # Compute normalized metrics
    pivot["online_ratio"] = pivot["Online"] / pivot["best_makespan"]
    pivot["naive_online_ratio"] = pivot["Naive Online"] / pivot["best_makespan"]
    pivot["naive_penalty"] = pivot["naive_online_ratio"] - pivot["online_ratio"]

    # Merge in component settings
    components = df[[
        "scheduler", "ranking_function", "append_only", "compare", "critical_path"
    ]].drop_duplicates()
    pivot = pd.merge(pivot, components, on="scheduler", how="left")

    # Remove outliers
    filtered = remove_outliers(pivot, ["online_ratio", "naive_penalty"])

    # Plot
    color_components = ["ranking_function", "append_only", "compare", "critical_path", "estimate_method"]
    sns.set(style="whitegrid", font_scale=1.2)

    print("Generating plots...")
    # for component in color_components:
    #     plt.figure(figsize=(12, 8))
    #     ax = sns.scatterplot(
    #         data=filtered,
    #         x="naive_penalty",
    #         y="online_ratio",
    #         hue=component,
    #         s=80,
    #         alpha=0.1,
    #         # no edgecolor to avoid clutter
    #         edgecolor=None,
    #     )

    #     ax.set_title(f"Online vs Best Normalization (Colored by {component})")
    #     ax.set_xlabel("Naive Penalty = (Naive - Online) / Best Makespan")
    #     ax.set_ylabel("Online / Best Makespan")
    #     plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    #     plt.axvline(0, color='gray', linestyle=':', linewidth=1)
    #     plt.legend(title=component, bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.tight_layout()

    #     out_path = OUTDIR / f"scatter_by_{component}_best_normalized.png"
    #     plt.savefig(out_path)
    #     plt.close()

    # Scheduler type comparison by workflow
    # print("Generating scheduler comparison...")
    
    # # Boxplot comparing scheduler types
    # plt.figure(figsize=(12, 8))
    # scheduler_comparison = pivot.melt(
    #     id_vars=['workflow'], 
    #     value_vars=['Offline', 'Online', 'Naive Online'],
    #     var_name='Scheduler_Type', 
    #     value_name='Makespan'
    # )
    # sns.boxplot(data=scheduler_comparison, x='workflow', y='Makespan', hue='Scheduler_Type')
    # plt.title('Scheduler Performance by Workflow')
    # plt.xlabel('Workflow')
    # plt.ylabel('Makespan')
    # plt.xticks(rotation=45)
    # plt.legend(title='Scheduler Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plt.savefig(OUTDIR / 'scheduler_comparison_by_workflow.png', dpi=300, bbox_inches='tight')
    # plt.close()


    # # Additional analysis
    # print("Generating extra analysis...")
    
    # # Average performance by CCR
    # plt.figure(figsize=(10, 6))
    # ccr_summary = pivot.groupby('ccr')[['online_ratio', 'naive_online_ratio']].mean()
    # ccr_summary.plot(kind='bar')
    # plt.title('Average Performance by CCR')
    # plt.ylabel('Ratio to Best Makespan')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(OUTDIR / 'average_performance_by_ccr.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    # # Performance variance by CCR
    # plt.figure(figsize=(10, 6))
    # ccr_variance = pivot.groupby('ccr')['online_ratio'].std()
    # ccr_variance.plot(kind='bar')
    # plt.title('Performance Variance by CCR')
    # plt.ylabel('Standard Deviation')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(OUTDIR / 'performance_variance_by_ccr.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # # Average performance by workflow
    # plt.figure(figsize=(12, 8))
    # workflow_summary = pivot.groupby('workflow')[['online_ratio', 'naive_online_ratio']].mean()
    # workflow_summary.plot(kind='bar')
    # plt.title('Average Performance by Workflow')
    # plt.ylabel('Ratio to Best Makespan')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(OUTDIR / 'average_performance_by_workflow.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    # # Performance variance by workflow
    # plt.figure(figsize=(12, 8))
    # workflow_variance = pivot.groupby('workflow')['online_ratio'].std()
    # workflow_variance.plot(kind='bar')
    # plt.title('Performance Variance by Workflow')
    # plt.ylabel('Standard Deviation')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(OUTDIR / 'performance_variance_by_workflow.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # plt.figure(figsize=(10, 8))
    # heatmap_data = pivot.pivot_table(
    #     index='workflow',
    #     columns='ccr',
    #     values='ratio_diff',
    #     aggfunc='mean'
    # )
    # sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', center=1.0, fmt='.3f')
    # plt.title(f'Ratio Difference Heatmap')
    # plt.tight_layout()
    # plt.savefig(OUTDIR / f'heatmap_ratio_diff.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    # # Component effectiveness by workflow
    # plt.figure(figsize=(12, 8))
    # component_workflow = pivot.groupby(['workflow', 'ranking_function'])['online_ratio'].mean().unstack()
    # component_workflow.plot(kind='bar')
    # plt.title('Component Effectiveness by Workflow')
    # plt.ylabel('Online/Best Ratio')
    # plt.xticks(rotation=45)
    # plt.legend(title='Ranking Function', bbox_to_anchor=(1.05, 1))
    # plt.tight_layout()
    # plt.savefig(OUTDIR / 'component_effectiveness_by_workflow.png', dpi=300, bbox_inches='tight')
    # plt.close()
 
    pivot["ratio_diff"] = pivot["online_ratio"] - pivot["naive_online_ratio"]

    components = ["ccr", "workflow", "ranking_function", "append_only", "compare", "critical_path", "estimate_method"]
    # sort by num unique values
    combos = list(combinations(sorted(components, key=lambda x: -pivot[x].nunique()), 2))
    fig, axes = plt.subplots(ncols=3, nrows=len(combos) // 3 + (len(combos) % 3 > 0), figsize=(18, 6 * (len(combos) // 3 + (len(combos) % 3 > 0))))
    axes: List[plt.Axes] = axes.flatten() if len(combos) > 1 else [axes]
    for i, (xcomp, ycomp) in enumerate(combos):
        print(f"Generating heatmap for {xcomp} vs {ycomp}...")
        heatmap_data = pivot.pivot_table(
            index=xcomp,
            columns=ycomp,
            values='ratio_diff',
            aggfunc='mean'
        )
        sns.heatmap(
            heatmap_data,
            annot=True, cmap='RdYlBu_r', center=0.0, fmt='.2f',
            vmin=-0.5, vmax=0.5,
            ax=axes[i] if len(combos) > 1 else axes[0]
        )

        axes[i].set_title(f'{xcomp} vs {ycomp}')
        axes[i].set_xlabel(ycomp)
        axes[i].set_ylabel(xcomp)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(OUTDIR / 'heatmap_ratio_diff_components.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Done. Plots saved to: {OUTDIR}")

if __name__ == "__main__":
    main()