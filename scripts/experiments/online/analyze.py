from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

# Setup paths
THISDIR = pathlib.Path(__file__).resolve().parent
CSV_PATH = THISDIR / "results.csv"
OUTDIR = THISDIR / "plots"
OUTDIR.mkdir(exist_ok=True)

def analyze():
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("CSV is empty. Exiting.")
        return

    # Pivot: (workflow, ccr, sample, variant, estimate_method) → scheduler_type → makespan
    pivot = df.pivot_table(
        index=["workflow", "ccr", "sample", "scheduler_variant", "estimate_method"],
        columns="scheduler_type",
        values="makespan"
    ).reset_index()

    # Keep only rows where all scheduler types are present
    pivot = pivot.dropna(subset=["Offline", "Online", "Naive Online"])

    # Normalize Online and Naive Online by Offline
    pivot["online_ratio"] = pivot["Online"] / pivot["Offline"]
    pivot["naive_online_ratio"] = pivot["Naive Online"] / pivot["Offline"]

    # Plot boxplots of normalized makespan ratios
    sns.set(style="whitegrid", font_scale=1.2)
    for estimate in sorted(pivot["estimate_method"].unique()):
        for variant in sorted(pivot["scheduler_variant"].unique()):
            df_subset = pivot[
                (pivot["estimate_method"] == estimate) &
                (pivot["scheduler_variant"] == variant)
            ]

            if df_subset.empty:
                continue

            # Online and Naive Online normalized boxplot
            melted = df_subset.melt(
                id_vars=["ccr", "workflow"],
                value_vars=["online_ratio", "naive_online_ratio"],
                var_name="Scheduler",
                value_name="Normalized Makespan"
            )
            scheduler_labels = {
                "online_ratio": "Online / Offline",
                "naive_online_ratio": "Naive Online / Offline"
            }
            melted["Scheduler"] = melted["Scheduler"].map(scheduler_labels)

            plt.figure(figsize=(12, 7))
            ax = sns.boxplot(
                data=melted,
                x="ccr",
                y="Normalized Makespan",
                hue="Scheduler",
                showfliers=False
            )
            ax.set_title(f"Normalized Makespan vs Offline ({estimate}, {variant})")
            ax.set_ylabel("Makespan Ratio (Online / Offline)")
            ax.set_xlabel("CCR")
            plt.legend(title="Scheduler")
            plt.tight_layout()

            out_path = OUTDIR / f"boxplot_normalized_makespan_{estimate}_{variant}.png"
            plt.savefig(out_path)
            plt.close()

    # Export summary stats of normalized ratios
    COL_NAMES = {
        "estimate_method": "Estimator",
        "scheduler_variant": "Scheduler",
        "online_ratio": "Online / Offline",
        "naive_online_ratio": "Naive Online / Offline",
    }
    summary_mean = pivot.groupby(["estimate_method", "scheduler_variant"])[
        ["online_ratio", "naive_online_ratio"]
    ].mean().reset_index()
    summary_std = pivot.groupby(["estimate_method", "scheduler_variant"])[
        ["online_ratio", "naive_online_ratio"]
    ].std().reset_index()
    # Merge mean and std into a single DataFrame with cell values <mean> \pm <std>
    summary = pd.merge(summary_mean, summary_std, on=["estimate_method", "scheduler_variant"], suffixes=("", "_std"))
    summary["online_ratio"] = summary.apply(
        lambda row: f"${row['online_ratio']:.3f} \pm {row['online_ratio_std']:.3f}$",
        axis=1
    )
    summary["naive_online_ratio"] = summary.apply(
        lambda row: f"${row['naive_online_ratio']:.3f} \pm {row['naive_online_ratio_std']:.3f}$",
        axis=1
    )
    summary = summary.rename(columns=COL_NAMES)
    summary = summary[["Estimator", "Scheduler", "Online / Offline", "Naive Online / Offline"]]

    print(summary.columns)
    summary.to_latex(
        OUTDIR / "normalized_makespan_stats.tex",
        index=False,
        escape=False,
        float_format="%.3f",
        label="tab:normalized_makespan_stats",
        caption="Normalized makespan statistics for online scheduling experiments."
    )

if __name__ == "__main__":
    analyze()
