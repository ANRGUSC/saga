import pathlib

from post_load import load_results_csv

thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    """Generate table of results."""
    df_results = load_results_csv()
    
    # remove all scheduler and base schedulers that start with "Not"
    df_results = df_results[~df_results["Scheduler"].str.startswith("Not")]
    df_results = df_results[~df_results["Base Scheduler"].str.startswith("Not")]

    best_scheduler = df_results.groupby(["Scheduler"])["Makespan Ratio"].max().idxmin()
    best_schduler_value = df_results.groupby(["Scheduler"])["Makespan Ratio"].max().min()
    print(f"Best scheduler: {best_scheduler} ({best_schduler_value:.2f})")

    df_results["Makespan Ratio"] = df_results["Makespan Ratio"].apply(lambda x: f"{x:.2f}" if x < 1e3 else ">1000")

    # pivot so base schedulers are columns and schedulers are rows
    df_results = df_results.pivot(index="Scheduler", columns="Base Scheduler", values="Makespan Ratio")
    df_results = df_results.reindex(sorted(df_results.columns), axis=1)
    df_results = df_results.reindex(sorted(df_results.index), axis=0)

    df_results = df_results.fillna("")
    df_results = df_results.replace("nan", "")

    (thisdir / "output").mkdir(exist_ok=True)

    # sort columns alphabetically
    df_results = df_results.reindex(sorted(df_results.columns), axis=1)
    
    df_results.columns = [f"\\textbf{{{col}}}" for col in df_results.columns]
    df_results.to_latex(
        buf=thisdir / "output" / "results.tex",
        escape=False,
        na_rep="N/A",
        bold_rows=True,
        caption="Makespan ratio for each scheduler relative to the base scheduler.",
        label="tab:sa_results"
    )

    # split into two sets of columns since the table is too wide
    num_cols = len(df_results.columns)
    df_results_split = [
        df_results.iloc[:, :num_cols//2],
        df_results.iloc[:, num_cols//2:]
    ]

    for i, _df_results in enumerate(df_results_split, start=1):
        _df_results.to_latex(
            buf=thisdir / "output" / f"results_{i}.tex",
            escape=False,
            na_rep="N/A",
            bold_rows=True,
            caption="Makespan ratio for each scheduler relative to the base scheduler.",
            label="tab:sa_results"
        )

if __name__ == "__main__":
    main()
