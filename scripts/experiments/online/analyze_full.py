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
    for component in color_components:
        plt.figure(figsize=(12, 8))
        ax = sns.scatterplot(
            data=filtered,
            x="naive_penalty",
            y="online_ratio",
            hue=component,
            s=80,
            alpha=0.1,
            # no edgecolor to avoid clutter
            edgecolor=None,
        )

        ax.set_title(f"Online vs Best Normalization (Colored by {component})")
        ax.set_xlabel("Naive Penalty = (Naive - Online) / Best Makespan")
        ax.set_ylabel("Online / Best Makespan")
        plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        plt.axvline(0, color='gray', linestyle=':', linewidth=1)
        plt.legend(title=component, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        out_path = OUTDIR / f"scatter_by_{component}_best_normalized.png"
        plt.savefig(out_path)
        plt.close()

    print(f"Done. Plots saved to: {OUTDIR}")

if __name__ == "__main__":
    main()
