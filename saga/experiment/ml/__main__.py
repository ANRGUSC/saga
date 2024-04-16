import argparse

from . import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare a dataset for machine learning.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_dataset_parser = subparsers.add_parser("prepare_dataset")
    prepare_dataset_parser.set_defaults(func=prepare_dataset)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
