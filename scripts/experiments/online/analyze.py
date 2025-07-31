from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

# Setup paths
THISDIR = pathlib.Path(__file__).resolve().parent
CSV_PATH = THISDIR / "results.csv"
OUTDIR = THISDIR / "plots"
OUTDIR.mkdir(exist_ok=True)
FILETYPE = "pdf"  # Change to "pdf" if needed

# Enable LATEX rendering in matplotlib
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # for \text command
plt.rcParams["font.serif"] = ["Computer Modern Roman"]

def analyze():
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("CSV is empty. Exiting.")
        return

    # Pivot: (workflow, ccr, sample, variant, estimate_method) → scheduler_type → makespan
    pivot = df.pivot_table(
        index=["workflow", "ccr", "sample", "scheduler", "estimate_method"],
        columns="scheduler_type",
        values="makespan"
    ).reset_index()

    # Keep only rows where all scheduler types are present
    pivot = pivot.dropna(subset=["Offline", "Online", "Naive Online"])

    # Normalize Online and Naive Online by Offline
    pivot["online_ratio"] = pivot["Online"] / pivot["Offline"]
    pivot["naive_online_ratio"] = pivot["Naive Online"] / pivot["Offline"]

    # Plot boxplots of normalized makespan ratios
    sns.set_theme(style="whitegrid", font_scale=1.6)
    for estimate in sorted(pivot["estimate_method"].unique()):
        for variant in sorted(pivot["scheduler"].unique()):
            df_subset = pivot[
                (pivot["estimate_method"] == estimate) &
                (pivot["scheduler"] == variant)
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
                "online_ratio": r"$\text{MR}_{\text{Online}}$",
                "naive_online_ratio": r"$\text{MR}_{\text{Naive}}$"
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
            ax.set_ylabel("Makespan Ratio")
            ax.set_xlabel("CCR")
            plt.legend(
                title="Scheduler",
                bbox_to_anchor=(1.05, 1),  # Moves legend to the right
                loc='upper left',
                borderaxespad=0.
            )
            plt.tight_layout()

            out_path = OUTDIR / f"boxplot_normalized_makespan_{estimate}_{variant}.{FILETYPE}"
            plt.savefig(out_path)
            plt.close()

    # Export summary stats of normalized ratios
    COL_NAMES = {
        "estimate_method": "Estimator",
        "scheduler": "Scheduler",
        "online_ratio": r"$\text{MR}_{\text{Online}}$",
        "naive_online_ratio": r"$\text{MR}_{\text{Naive}}$"
    }
    summary_mean = pivot.groupby(["estimate_method", "scheduler"])[
        ["online_ratio", "naive_online_ratio"]
    ].mean().reset_index()
    summary_std = pivot.groupby(["estimate_method", "scheduler"])[
        ["online_ratio", "naive_online_ratio"]
    ].std().reset_index()
    # Merge mean and std into a single DataFrame with cell values <mean> \pm <std>
    summary = pd.merge(summary_mean, summary_std, on=["estimate_method", "scheduler"], suffixes=("", "_std"))
    summary["online_ratio"] = summary.apply(
        lambda row: f"${row['online_ratio']:.3f} \pm {row['online_ratio_std']:.3f}$",
        axis=1
    )
    summary["naive_online_ratio"] = summary.apply(
        lambda row: f"${row['naive_online_ratio']:.3f} \pm {row['naive_online_ratio_std']:.3f}$",
        axis=1
    )
    summary = summary.rename(columns=COL_NAMES)
    summary = summary[[col for col in COL_NAMES.values() if col in summary.columns]]

    summary.to_latex(
        OUTDIR / "normalized_makespan_stats.tex",
        index=False,
        escape=False,
        float_format="%.3f",
        label="tab:normalized_makespan_stats",
        caption="Normalized makespan statistics for online scheduling experiments.",
        position="ht!"
    )


    # Multi-plot (one subplot per workflow) with shared y-axis
    workflows = sorted(pivot["workflow"].unique())
    n_workflows = len(workflows)
    ncols = 2  # You can adjust based on your desired layout
    nrows = -(-n_workflows // ncols)  # Ceiling division

    for estimate in sorted(pivot["estimate_method"].unique()):
        for variant in sorted(pivot["scheduler"].unique()):
            df_subset = pivot[
                (pivot["estimate_method"] == estimate) &
                (pivot["scheduler"] == variant)
            ]
            if df_subset.empty:
                continue

            melted = df_subset.melt(
                id_vars=["ccr", "workflow"],
                value_vars=["online_ratio", "naive_online_ratio"],
                var_name="Scheduler",
                value_name="Normalized Makespan"
            )
            scheduler_labels = {
                "online_ratio": r"$\text{MR}_{\text{Online}}$",
                "naive_online_ratio": r"$\text{MR}_{\text{Naive}}$"
            }
            melted["Scheduler"] = melted["Scheduler"].map(scheduler_labels)

            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows), sharey=True
            )
            axes: List[plt.Axes] = axes.flatten()
            
            fig.delaxes(axes[1])  # Remove the second subplot
            axes = [axes[0], *axes[2:]] # Skip the second subplot for aesthetics
            
            for i, workflow in enumerate(workflows):
                ax = axes[i]
                data = melted[melted["workflow"] == workflow]
                sns.boxplot(
                    data=data,
                    x="ccr",
                    y="Normalized Makespan",
                    hue="Scheduler",
                    showfliers=False,
                    ax=ax
                )
                ax.set_title(workflow)
                ax.set_xlabel("CCR")
                if i % ncols == 0:
                    ax.set_ylabel("Normalized Makespan")
                else:
                    ax.set_ylabel("")

                ax.get_legend().remove()

            # Add a single shared legend outside the figure
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles, labels,
                title="Scheduler",
                loc='center right',
                bbox_to_anchor=(0.835, 0.85),
                borderaxespad=0.
            )

            fig.suptitle(f"Normalized Makespan by Workflow ({estimate}, {variant})")
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            out_path = OUTDIR / f"subplot_normalized_makespan_{estimate}_{variant}.{FILETYPE}"
            fig.savefig(out_path)
            plt.close(fig)


if __name__ == "__main__":
    analyze()
