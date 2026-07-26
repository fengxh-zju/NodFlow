import argparse
import json
from pathlib import Path

from src.utils.io import ensure_dir, load_json, save_json, write_jsonl


METHOD_ID = "nodflow"
METHOD_SCHEMA = {
    "uses": [
        "normal_source_roi",
        "target_mask",
        "target_histogram",
        "frozen_volumetric_prior",
    ],
    "note": "NodFlow input manifest for MAISI or CTFlow.",
}
SEMANTIC_FIELDS = (
    "texture",
    "calcification",
    "malignancy",
    "margin",
    "sphericity",
    "spiculation",
    "subtlety",
    "lobulation",
)


def _read_split(root, split, splits_dir=None):
    path = Path(splits_dir or Path(root) / "splits") / f"{split}_cases.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _relative(path, root):
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def _sample_paths(root, sample_id, kind):
    base = Path(root) / "rois" / kind
    return {
        "image": base / "images" / f"{sample_id}.nii.gz",
        "mask": base / "masks" / f"{sample_id}.nii.gz",
        "metadata": base / "metadata" / f"{sample_id}.json",
    }


def _row(root, sample_id, split):
    pathological = _sample_paths(root, sample_id, "pathological")
    normal = _sample_paths(root, sample_id, "normal")
    metadata = load_json(pathological["metadata"]) if pathological["metadata"].exists() else {}
    condition = {
        "subtype": metadata.get("subtype", "unknown"),
        "diameter_mm": metadata.get("diameter_mm"),
        "histogram_source": _relative(pathological["metadata"], root),
    }
    condition.update(
        {key: metadata[key] for key in SEMANTIC_FIELDS if metadata.get(key) is not None}
    )
    return {
        "sample_id": sample_id,
        "split": split,
        "case_id": metadata.get("case_id"),
        "nodule_id": metadata.get("nodule_id"),
        "subtype": metadata.get("subtype", "unknown"),
        "diameter_mm": metadata.get("diameter_mm"),
        "source_dataset": metadata.get("source_dataset"),
        "source_image": _relative(normal["image"], root),
        "source_mask": _relative(normal["mask"], root),
        "target_image": _relative(pathological["image"], root),
        "target_mask": _relative(pathological["mask"], root),
        "pathological_metadata": _relative(pathological["metadata"], root),
        "condition": condition,
    }


def build_manifests(processed_root, output_root, splits, splits_dir=None):
    processed_root = Path(processed_root)
    output_root = Path(output_root)
    method_dir = output_root / METHOD_ID
    ensure_dir(method_dir)
    save_json(METHOD_SCHEMA, method_dir / "schema.json")

    summary = {
        "processed_root": str(processed_root),
        "splits_dir": str(Path(splits_dir or processed_root / "splits")),
        "methods": {METHOD_ID: {}},
        "splits": splits,
    }
    for split in splits:
        rows = [
            _row(processed_root, sample_id, split)
            for sample_id in _read_split(processed_root, split, splits_dir=splits_dir)
        ]
        write_jsonl(rows, method_dir / f"{split}.jsonl")
        summary["methods"][METHOD_ID][split] = len(rows)
    save_json(summary, output_root / "manifest_summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build NodFlow manifests from processed ROI splits.")
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()
    output_root = args.output_root or str(Path(args.processed_root) / "manifests")
    summary = build_manifests(
        args.processed_root,
        output_root,
        args.splits,
        splits_dir=args.splits_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
