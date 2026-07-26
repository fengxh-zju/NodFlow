import argparse

from src.nodflow.method import NodFlow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_config", required=True)
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    NodFlow(args.method_config, args.data_config).train(args.output_dir)


if __name__ == "__main__":
    main()
