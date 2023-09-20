from post_load import load_results_csv


def main():
    df_results = load_results_csv()

    df_hybrid = df_results[df_results["Scheduler"].str.startswith("Not")]
    for row in df_hybrid.itertuples():
        print(row)


if __name__ == "__main__":
    main()
