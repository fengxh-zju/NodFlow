import argparse

from src.nodflow.method import NodFlow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_config", required=True)
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--target_library", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    NodFlow(args.method_config, args.data_config).generate(
        checkpoint=args.checkpoint,
        target_library=args.target_library,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
