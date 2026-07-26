import argparse
from pathlib import Path

from src.nodflow.energies import load_posterior_energies
from src.utils.config import load_config, load_method_config


DATA_REQUIRED = {
    "dataset",
    "root",
    "output_root",
    "target_library",
    "splits",
    "generation_split",
}
METHOD_REQUIRED = {"method_id", "implementation", "prior"}


def _placeholder_paths(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _placeholder_paths(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _placeholder_paths(item, f"{prefix}[{index}]")
    elif isinstance(value, str) and (
        value.startswith("<")
        or value.endswith(">")
        or "${" in value
        or value.strip().upper() in {"REQUIRED", "CHOOSE_ME", "TBD"}
    ):
        yield prefix


def validate(path, kind):
    config = load_config(path)
    required = set(DATA_REQUIRED if kind == "data" else METHOD_REQUIRED)
    if kind == "method":
        required.add("hyperparameters_file")

    missing = sorted(key for key in required if key not in config or config[key] in (None, ""))
    if missing:
        raise ValueError(f"{kind} config is missing required keys: {', '.join(missing)}")

    unresolved = sorted(_placeholder_paths(config))
    if unresolved:
        raise ValueError(
            "configuration still contains unresolved placeholders at: "
            + ", ".join(unresolved)
        )
    if kind == "method":
        merged = load_method_config(path)
        load_posterior_energies(merged, str(merged["prior"]))
    return config


def main():
    parser = argparse.ArgumentParser(description="Validate a public runtime configuration.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--kind", choices=["data", "method"], required=True)
    args = parser.parse_args()
    config = validate(args.config, args.kind)
    print(
        f"valid {args.kind} config: {args.config} "
        f"({len(config)} top-level keys)"
    )


if __name__ == "__main__":
    main()
