import argparse
import json
from pathlib import Path

from src.utils.config import load_config
from src.utils.io import load_json, save_json


METHOD_REQUIRED_FIELDS = {
    "nodflow": ["source_image", "target_mask", "pathological_metadata"],
}

GROUP_KEYS = ("patient_id", "case_id", "series_uid", "source_series_dir")


def _read_ids(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    return rows


def _check_path(root, rel, label, failures):
    if rel in [None, ""]:
        failures.append(f"missing {label}")
        return
    path = Path(rel)
    if not path.is_absolute():
        path = Path(root) / path
    if not path.exists():
        failures.append(f"missing {label}: {path}")


def _sample_metadata(root, sample_id):
    candidates = [
        Path(root) / "metadata" / f"{sample_id}.json",
        Path(root) / "rois" / "pathological" / "metadata" / f"{sample_id}.json",
    ]
    for path in candidates:
        if path.is_file():
            return load_json(path), path
    return None, candidates[0]


def _metadata_group(payload, sample_id):
    for key in GROUP_KEYS:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return str(sample_id)


def validate_splits(root, splits_dir):
    splits = {}
    failures = []
    for split in ["train", "val", "test"]:
        path = Path(splits_dir) / f"{split}_cases.txt"
        try:
            ids = _read_ids(path)
        except FileNotFoundError:
            failures.append(f"missing split file: {path}")
            ids = []
        splits[split] = ids
        for sid in ids:
            for rel in [
                f"rois/pathological/images/{sid}.nii.gz",
                f"rois/pathological/masks/{sid}.nii.gz",
                f"rois/pathological/metadata/{sid}.json",
            ]:
                if not (Path(root) / rel).exists():
                    failures.append(f"split {split} references missing ROI file: {rel}")
    seen = {}
    for split, ids in splits.items():
        for sid in ids:
            if sid in seen:
                failures.append(f"sample leakage: {sid} appears in {seen[sid]} and {split}")
            seen[sid] = split
    derived_groups = {}
    group_owner = {}
    for split, ids in splits.items():
        split_groups = set()
        for sid in ids:
            metadata, metadata_path = _sample_metadata(root, sid)
            if metadata is None:
                failures.append(f"missing grouping metadata for {sid}: {metadata_path}")
                continue
            group = _metadata_group(metadata, sid)
            split_groups.add(group)
            if group in group_owner and group_owner[group] != split:
                failures.append(
                    f"metadata group leakage: {group} appears in {group_owner[group]} and {split}"
                )
            group_owner[group] = split
        derived_groups[split] = split_groups
    for split in ["train", "val", "test"]:
        group_path = Path(splits_dir) / f"{split}_groups.txt"
        if group_path.exists():
            declared_groups = set(_read_ids(group_path))
            for group in declared_groups:
                key = f"group:{group}"
                if key in seen:
                    failures.append(f"group leakage: {group} appears in {seen[key]} and {split}")
                seen[key] = split
            if declared_groups != derived_groups.get(split, set()):
                failures.append(
                    f"{split}_groups.txt does not match metadata-derived groups: "
                    f"declared={len(declared_groups)} derived={len(derived_groups.get(split, set()))}"
                )
        else:
            failures.append(f"missing split group file: {group_path}")
    summary_path = Path(splits_dir) / "split_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        for split, ids in splits.items():
            expected = summary.get("splits", {}).get(split, {}).get("num_samples")
            if expected is not None and int(expected) != len(ids):
                failures.append(f"split_summary mismatch for {split}: {expected} != {len(ids)}")
    else:
        failures.append(f"missing split summary: {summary_path}")
    return splits, failures


def validate_target_library(target_root, train_ids):
    target_root = Path(target_root)
    failures = []
    rows = []
    meta_dir = target_root / "metadata"
    if not meta_dir.exists():
        return rows, [f"missing target library metadata dir: {meta_dir}"]
    for meta_path in sorted(meta_dir.glob("*.json")):
        meta = load_json(meta_path)
        rows.append(meta)
        sid = meta.get("source_sample_id")
        if not sid:
            failures.append(f"{meta_path} missing source_sample_id")
        elif sid not in train_ids:
            failures.append(f"target library leakage: {meta_path.name} source_sample_id={sid} is not in train split")
        _check_path(target_root, meta.get("mask_path"), f"{meta_path.name}.mask_path", failures)
        _check_path(target_root, meta.get("hist_path"), f"{meta_path.name}.hist_path", failures)
    if not rows:
        failures.append(f"empty target library: {meta_dir}")
    return rows, failures


def validate_manifests(root, manifests_dir, splits, methods):
    manifests_dir = Path(manifests_dir)
    failures = []
    counts = {}
    if not manifests_dir.exists():
        return counts, [f"missing manifests dir: {manifests_dir}"]
    for method in methods:
        method_dir = manifests_dir / method
        schema_path = method_dir / "schema.json"
        if not schema_path.exists():
            failures.append(f"missing manifest schema: {schema_path}")
        counts[method] = {}
        required = METHOD_REQUIRED_FIELDS.get(method, [])
        for split, split_ids in splits.items():
            split_set = set(split_ids)
            path = method_dir / f"{split}.jsonl"
            if not path.exists():
                failures.append(f"missing manifest: {path}")
                counts[method][split] = 0
                continue
            rows = _read_jsonl(path)
            counts[method][split] = len(rows)
            if len(rows) != len(split_ids):
                failures.append(f"{method}/{split} row count {len(rows)} != split count {len(split_ids)}")
            for row in rows:
                sid = row.get("sample_id")
                if sid not in split_set:
                    failures.append(f"{method}/{split} row sample_id={sid} is not in {split}_cases.txt")
                if row.get("split") != split:
                    failures.append(f"{method}/{split} row sample_id={sid} has split={row.get('split')}")
                for field in required:
                    _check_path(root, row.get(field), f"{method}/{split}/{sid}.{field}", failures)
    return counts, failures


def main():
    parser = argparse.ArgumentParser(description="Validate leakage-safe real-data splits, target library, and method manifests.")
    parser.add_argument("--data_config", default=None)
    parser.add_argument("--processed_root", default=None)
    parser.add_argument("--target_library", default=None)
    parser.add_argument("--splits_dir", default=None)
    parser.add_argument("--manifests_dir", default=None)
    parser.add_argument("--methods", nargs="+", default=sorted(METHOD_REQUIRED_FIELDS))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.data_config) if args.data_config else {}
    root_value = args.processed_root or cfg.get("root")
    if not root_value:
        raise ValueError("provide --data_config or --processed_root")
    root = Path(root_value)
    splits_dir = Path(args.splits_dir or cfg.get("splits", root / "splits"))
    target_library = Path(args.target_library or cfg.get("target_library", root / "target_library"))
    manifests_dir = Path(args.manifests_dir or cfg.get("manifests", root / "manifests"))

    splits, split_failures = validate_splits(root, splits_dir)
    target_rows, target_failures = validate_target_library(target_library, set(splits.get("train", [])))
    manifest_counts, manifest_failures = validate_manifests(root, manifests_dir, splits, args.methods)

    failures = split_failures + target_failures + manifest_failures
    result = {
        "status": "pass" if not failures else "fail",
        "processed_root": str(root),
        "splits_dir": str(splits_dir),
        "target_library": str(target_library),
        "manifests_dir": str(manifests_dir),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "target_library_count": len(target_rows),
        "manifest_counts": manifest_counts,
        "failures": failures,
    }
    if args.output:
        save_json(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
