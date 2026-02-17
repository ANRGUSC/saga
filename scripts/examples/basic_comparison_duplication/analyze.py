import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
from matplotlib.patches import Patch

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

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
    
    # Clean up old plot files (PNG only - keep CSV files)
    print("Cleaning up old plots...")
    old_files = list(glob.glob(str(thisdir / "*.png")))
    
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
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = []
    labels = []
    data_groups = []
    colors = []
    
    for i, scheduler in enumerate(['HEFT', 'CPoP']):
        for j, dup in enumerate(['1', '2']):
            subset = df_clean[(df_clean['Scheduler'] == scheduler) & (df_clean['Dup Factor'] == dup)]
            positions.append(i * 3 + j)
            labels.append(f"{scheduler}\nDup={dup}")
            data_groups.append(subset['Makespan'].values)
            colors.append('lightblue' if dup == '1' else 'orange')
    
    bp = ax.boxplot(data_groups, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black', linewidth=2))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Makespan', fontsize=12)
    ax.set_title('Effect of Duplication on Makespan', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [Patch(facecolor='lightblue', label='No Duplication (Factor=1)'),
                      Patch(facecolor='orange', label='With Duplication (Factor=2)')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(thisdir / "duplication_overall.png")
    plt.close()
    
    # 2. CCR effect with duplication
    schedulers = ['HEFT', 'CPoP']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, scheduler in enumerate(schedulers):
        ax = axes[idx]
        df_sched = df_clean[df_clean['Scheduler'] == scheduler]
        
        ccr_values = sorted(df_sched['CCR'].unique())
        positions = []
        data_groups = []
        colors = []
        
        for i, ccr in enumerate(ccr_values):
            for j, dup in enumerate(['1', '2']):
                subset = df_sched[(df_sched['CCR'] == ccr) & (df_sched['Dup Factor'] == dup)]
                positions.append(i * 3 + j)
                data_groups.append(subset['Makespan'].values)
                colors.append('lightblue' if dup == '1' else 'orange')
        
        bp = ax.boxplot(data_groups, positions=positions, widths=0.6, patch_artist=True,
                       showfliers=False, medianprops=dict(color='black', linewidth=2))
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        tick_positions = [i * 3 + 0.5 for i in range(len(ccr_values))]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{ccr}" for ccr in ccr_values], fontsize=10)
        ax.set_xlabel('CCR', fontsize=11)
        ax.set_ylabel('Makespan', fontsize=11)
        ax.set_title(f'{scheduler}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    legend_elements = [Patch(facecolor='lightblue', label='Dup Factor=1'),
                      Patch(facecolor='orange', label='Dup Factor=2')]
    axes[0].legend(handles=legend_elements, loc='upper left')
    
    fig.suptitle('CCR Impact on Makespan (with/without Duplication)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(thisdir / "ccr_effect.png")
    plt.close()
    
    # 3. Number of nodes effect with duplication
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, scheduler in enumerate(schedulers):
        ax = axes[idx]
        df_sched = df_clean[df_clean['Scheduler'] == scheduler]
        
        num_nodes_values = sorted(df_sched['Num Nodes'].unique())
        positions = []
        data_groups = []
        colors = []
        
        for i, num_nodes in enumerate(num_nodes_values):
            for j, dup in enumerate(['1', '2']):
                subset = df_sched[(df_sched['Num Nodes'] == num_nodes) & (df_sched['Dup Factor'] == dup)]
                positions.append(i * 3 + j)
                data_groups.append(subset['Makespan'].values)
                colors.append('lightblue' if dup == '1' else 'orange')
        
        bp = ax.boxplot(data_groups, positions=positions, widths=0.6, patch_artist=True,
                       showfliers=False, medianprops=dict(color='black', linewidth=2))
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        tick_positions = [i * 3 + 0.5 for i in range(len(num_nodes_values))]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{n}" for n in num_nodes_values], fontsize=10)
        ax.set_xlabel('Number of Nodes', fontsize=11)
        ax.set_ylabel('Makespan', fontsize=11)
        ax.set_title(f'{scheduler}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    legend_elements = [Patch(facecolor='lightblue', label='Dup Factor=1'),
                      Patch(facecolor='orange', label='Dup Factor=2')]
    axes[0].legend(handles=legend_elements, loc='upper left')
    
    fig.suptitle('Network Size Impact on Makespan (with/without Duplication)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(thisdir / "num_nodes_effect.png")
    plt.close()
    
    # 4. Branching factor effect with duplication
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, scheduler in enumerate(schedulers):
        ax = axes[idx]
        df_sched = df_clean[df_clean['Scheduler'] == scheduler]
        
        branching_values = sorted(df_sched['Branching Factor'].unique())
        positions = []
        data_groups = []
        colors = []
        
        for i, branching in enumerate(branching_values):
            for j, dup in enumerate(['1', '2']):
                subset = df_sched[(df_sched['Branching Factor'] == branching) & (df_sched['Dup Factor'] == dup)]
                positions.append(i * 3 + j)
                data_groups.append(subset['Makespan'].values)
                colors.append('lightblue' if dup == '1' else 'orange')
        
        bp = ax.boxplot(data_groups, positions=positions, widths=0.6, patch_artist=True,
                       showfliers=False, medianprops=dict(color='black', linewidth=2))
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        tick_positions = [i * 3 + 0.5 for i in range(len(branching_values))]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{b}" for b in branching_values], fontsize=10)
        ax.set_xlabel('Branching Factor', fontsize=11)
        ax.set_ylabel('Makespan', fontsize=11)
        ax.set_title(f'{scheduler}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    legend_elements = [Patch(facecolor='lightblue', label='Dup Factor=1'),
                      Patch(facecolor='orange', label='Dup Factor=2')]
    axes[0].legend(handles=legend_elements, loc='upper left')
    
    fig.suptitle('Branching Factor Impact on Makespan (with/without Duplication)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(thisdir / "branching_effect.png")
    plt.close()
    
    # 5. Levels effect with duplication
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, scheduler in enumerate(schedulers):
        ax = axes[idx]
        df_sched = df_clean[df_clean['Scheduler'] == scheduler]
        
        levels_values = sorted(df_sched['Levels'].unique())
        positions = []
        data_groups = []
        colors = []
        
        for i, levels in enumerate(levels_values):
            for j, dup in enumerate(['1', '2']):
                subset = df_sched[(df_sched['Levels'] == levels) & (df_sched['Dup Factor'] == dup)]
                positions.append(i * 3 + j)
                data_groups.append(subset['Makespan'].values)
                colors.append('lightblue' if dup == '1' else 'orange')
        
        bp = ax.boxplot(data_groups, positions=positions, widths=0.6, patch_artist=True,
                       showfliers=False, medianprops=dict(color='black', linewidth=2))
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        tick_positions = [i * 3 + 0.5 for i in range(len(levels_values))]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{l}" for l in levels_values], fontsize=10)
        ax.set_xlabel('Levels', fontsize=11)
        ax.set_ylabel('Makespan', fontsize=11)
        ax.set_title(f'{scheduler}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    legend_elements = [Patch(facecolor='lightblue', label='Dup Factor=1'),
                      Patch(facecolor='orange', label='Dup Factor=2')]
    axes[0].legend(handles=legend_elements, loc='upper left')
    
    fig.suptitle('Graph Depth (Levels) Impact on Makespan (with/without Duplication)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(thisdir / "levels_effect.png")
    plt.close()
    
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
    
    # Improvement summary plot
    fig, ax = plt.subplots(figsize=(8, 6))
    summary_data = df_wide.groupby("Scheduler")["Improvement %"].mean().reset_index()
    
    bars = ax.bar(summary_data["Scheduler"], summary_data["Improvement %"], 
                  color=['steelblue', 'coral'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=11, fontweight='bold')
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Scheduler', fontsize=12)
    ax.set_ylabel('Avg % Improvement', fontsize=12)
    ax.set_title('Average Makespan Improvement from Duplication', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(thisdir / "improvement_summary.png")
    plt.close()
    
    # Save detailed improvement stats
    improvement_stats = df_wide.groupby("Scheduler").agg({
        "Improvement %": ["mean", "std", "min", "max"]
    }).round(2)
    improvement_stats.to_csv(thisdir / "improvement_statistics.csv")
    
    print(f"\n✓ Generated 6 plots (PNG only):")
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
