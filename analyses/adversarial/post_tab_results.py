import pathlib
import pickle
from typing import Dict

import pandas as pd
from simulated_annealing import SimulatedAnnealing
from io import StringIO

thisdir = pathlib.Path(__file__).parent.absolute()

def load_results() -> Dict[str, Dict[str, SimulatedAnnealing]]:
    results = {}
    for base_path in (thisdir / "results").glob("*"):
        results[base_path.name] = {}
        for path in base_path.glob("*.pkl"):
            results[base_path.name][path.stem] = pickle.loads(path.read_bytes())
    return results

def main():
    """Generate table of results."""
    keep_schedulers = [
        # "CPOP",
        # "HEFT",
        # "Duplex",
        # "MaxMin",
        # "MinMin",
        # # "FCP",
        # "MET",
        # "MCT",
    ]
    results = load_results()

    rows = []
    for base_scheduler_name, base_scheduler_results in results.items():
        if keep_schedulers and not base_scheduler_name in keep_schedulers:
            continue
        for scheduler_name, scheduler_results in base_scheduler_results.items():
            if keep_schedulers and not scheduler_name in keep_schedulers:
                continue
            makespan_ratio = scheduler_results.iterations[-1].best_energy
            if makespan_ratio > 1e3:
                makespan_ratio = ">1000"
            else:
                makespan_ratio = f"{makespan_ratio:.2f}"
            rows.append({
                "Base Scheduler": base_scheduler_name,
                "Scheduler": scheduler_name,
                "Makespan Ratio": makespan_ratio
            })

    df_results = pd.DataFrame(rows)
    # pivot so base schedulers are columns and schedulers are rows
    df_results = df_results.pivot(index="Scheduler", columns="Base Scheduler", values="Makespan Ratio")
    df_results = df_results.reindex(sorted(df_results.columns), axis=1)
    df_results = df_results.reindex(sorted(df_results.index), axis=0)

    # # get max value for each row
    # df_results["Max"] = df_results.max(axis=1, skipna=True, numeric_only=True)

    df_results = df_results.fillna("")
    df_results = df_results.replace("nan", "")

    (thisdir / "output").mkdir(exist_ok=True)
    df_results.to_csv(thisdir / "output" / "results.csv")
    df_results.to_string(buf=thisdir / "output" / "results.txt")

    df_results.to_latex(
        buf=thisdir / "output" / "results.tex",
        escape=False,
        column_format="|c|c|c|c|c|c|c|c|c|c|c|c|c|c|",
        multicolumn_format="c",
        multicolumn=True,
        na_rep="N/A",
        bold_rows=True,
        caption="Makespan ratio for each scheduler relative to the base scheduler.",
        label="tab:sa_results"
    )

    # Highlight max value in each row
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #dc3545' if v else '' for v in is_max]

    # Apply the function to the DataFrame
    styled_df = df_results.style.apply(highlight_max, axis=1)
    styled_df.set_table_attributes('class="table table-striped table-hover"')

    # Add styling to the table
    styled_df.set_table_styles([
        {"selector": "th", "props": [("text-align", "center")]},
        {"selector": "td", "props": [("text-align", "center")]},
        {"selector": "caption", "props": [("caption-side", "bottom")]}
    ])

    # Generate the HTML representation with Bootstrap classes
    html_buffer = StringIO()
    styled_df.to_html(
        buf=html_buffer,
        escape=False,
        classes="table table-striped table-hover table-sm",
        index_names=False,
        justify="center"
    )

    html_buffer.seek(0)
    base_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            <!-- set font size smaller for all elements -->
            * {{
                font-size: 0.8rem;
            }}
        </style>
    </head>
    <body>
        {html_buffer.read()}
    </body>
    </html>
    """

    (thisdir / "output" / "results.html").write_text(base_html)

if __name__ == "__main__":
    main()
