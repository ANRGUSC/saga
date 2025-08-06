import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

# Setup paths
THISDIR = pathlib.Path(__file__).resolve().parent
CSV_PATH = THISDIR / "workflow_variances.csv"
SAVEDIR = THISDIR / "plots-workflow-stats"
SAVEDIR.mkdir(exist_ok=True)

# Load the data
df = pd.read_csv(CSV_PATH)

# --- Summary Stats ---

def summarize_dataframe(df: pd.DataFrame):
    print("\n--- Global Summary ---")
    print(df.describe())

    print("\n--- Per-Workflow Summary (Mean) ---")
    grouped = df.groupby("workflow").mean(numeric_only=True)
    print(grouped)

    print("\n--- Per-Workflow Summary (Std) ---")
    print(df.groupby("workflow").std(numeric_only=True))


# --- Visualization Functions ---

def plot_bar_means(df: pd.DataFrame, stat_col: str, ylabel: str):
    means = df.groupby("workflow")[stat_col].mean()
    means = means.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=means.index, y=means.values)
    plt.ylabel(ylabel)
    plt.title(f"Mean {ylabel} per Workflow")
    plt.xticks(rotation=45)
    plt.tight_layout()
    outpath = SAVEDIR / f"barplot_{stat_col}.png"
    plt.savefig(outpath)
    plt.close()


def plot_violin_distribution(df: pd.DataFrame, stat_col: str, ylabel: str):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="workflow", y=stat_col, inner="box")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Distribution per Workflow")
    plt.xticks(rotation=45)
    plt.tight_layout()
    outpath = SAVEDIR / f"violin_{stat_col}.png"
    plt.savefig(outpath)
    plt.close()


def main():
    summarize_dataframe(df)

    # Key metrics to plot
    metrics = [
        ("task_variance_mean", "Task Variance (Mean)"),
        ("dep_variance_max", "Dependency Variance (Max)"),
        ("node_variance_mean", "Node Variance (Mean)"),
        ("link_variance_max", "Link Variance (Max)")
    ]

    for stat_col, ylabel in metrics:
        plot_bar_means(df, stat_col, ylabel)
        plot_violin_distribution(df, stat_col, ylabel)

    print(f"\nSaved all plots to: {SAVEDIR.resolve()}")


if __name__ == "__main__":
    main()
