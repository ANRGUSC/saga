import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

# Setup paths
THISDIR = pathlib.Path(__file__).resolve().parent
CSV_PATH = THISDIR / "results.csv"
OUTDIR = THISDIR / "analysis_plots"
OUTDIR.mkdir(exist_ok=True)

def analyze():
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("CSV is empty. Exiting.")
        return

    # Pivot: (workflow, ccr, sample, variant) → scheduler_type → makespan
    pivot = df.pivot_table(
        index=["workflow", "ccr", "sample", "scheduler_variant"],
        columns="scheduler_type",
        values="makespan"
    ).reset_index()

    # Keep only rows where both Online and Naive Online exist
    pivot = pivot.dropna(subset=["Online", "Naive Online"])

    # Compute metrics
    pivot["relative_improvement_pct"] = (
        (pivot["Naive Online"] - pivot["Online"]) / pivot["Naive Online"]
    ) * 100
    pivot["speedup"] = pivot["Naive Online"] / pivot["Online"]

    # Plot one boxplot per scheduler variant
    sns.set(style="whitegrid", font_scale=1.2)
    for variant in sorted(pivot["scheduler_variant"].unique()):
        df_variant = pivot[pivot["scheduler_variant"] == variant]

        plt.figure(figsize=(12, 7))
        ax = sns.boxplot(
            data=df_variant,
            x="ccr",
            y="relative_improvement_pct",
            hue="workflow",
            showfliers=False
        )
        ax.set_title(f"Relative Improvement (%) of Online vs Naive Online — {variant}")
        ax.set_ylabel("Relative Improvement (%)")
        ax.set_xlabel("CCR")
        ax.legend(title="Workflow")
        plt.tight_layout()

        out_path = OUTDIR / f"boxplot_relative_improvement_{variant}.png"
        plt.savefig(out_path)
        plt.close()

    # Optional: print some stats
    print("\n=== Summary: Relative Improvement by Scheduler Variant ===")
    print(pivot.groupby("scheduler_variant")["relative_improvement_pct"].describe())

if __name__ == "__main__":
    analyze()
