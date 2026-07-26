import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from src.utils.io import ensure_dir, load_json, save_json


def _stable_group(meta, group_by):
    for key in group_by:
        value = meta.get(key)
        if value not in [None, ""]:
            return str(value)
    return str(meta.get("sample_id") or meta.get("target_id") or "")


def _stable_sort_key(value, seed):
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return digest


def load_grouped_samples(metadata_dir, group_by):
    grouped = defaultdict(list)
    metadata_dir = Path(metadata_dir)
    for path in sorted(metadata_dir.glob("*.json")):
        meta = load_json(path)
        sid = str(meta.get("sample_id") or path.stem)
        group = _stable_group(meta, group_by) or sid
        grouped[group].append(sid)
    return {group: sorted(ids) for group, ids in grouped.items()}


def assign_groups(groups, train, val, test, seed):
    if not groups:
        return {"train": [], "val": [], "test": []}
    total = train + val + test
    if total <= 0:
        raise ValueError("split fractions must sum to a positive value")
    train = train / total
    val = val / total
    groups = sorted(groups, key=lambda item: _stable_sort_key(item, seed))
    n = len(groups)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    if n >= 3:
        n_train = min(max(1, n_train), n - 2)
        n_val = min(max(1, n_val), n - n_train - 1)
    elif n == 2:
        n_train = 1
        n_val = 0
    else:
        n_train = 1
        n_val = 0
    return {
        "train": groups[:n_train],
        "val": groups[n_train : n_train + n_val],
        "test": groups[n_train + n_val :],
    }


def write_splits(grouped, assignments, output_dir):
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    summary = {
        "num_groups": sum(len(groups) for groups in assignments.values()),
        "num_samples": sum(len(ids) for ids in grouped.values()),
        "splits": {},
    }
    seen_groups = {}
    for split, groups in assignments.items():
        ids = []
        for group in groups:
            if group in seen_groups:
                raise ValueError(f"group leakage: {group} assigned to {seen_groups[group]} and {split}")
            seen_groups[group] = split
            ids.extend(grouped[group])
        ids = sorted(ids)
        (output_dir / f"{split}_cases.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        (output_dir / f"{split}_groups.txt").write_text(
            "\n".join(groups) + ("\n" if groups else ""),
            encoding="utf-8",
        )
        summary["splits"][split] = {"num_groups": len(groups), "num_samples": len(ids)}
    save_json(summary, output_dir / "split_summary.json")
    return summary


def build_splits(metadata_dir, output_dir, train=0.7, val=0.1, test=0.2, group_by=None, seed=42):
    group_by = group_by or ["patient_id", "case_id", "series_uid", "source_series_dir"]
    grouped = load_grouped_samples(metadata_dir, group_by)
    assignments = assign_groups(list(grouped), train, val, test, seed)
    return write_splits(grouped, assignments, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Build leakage-safe train/val/test splits from ROI metadata.")
    parser.add_argument("--metadata_dir", required=True)
    parser.add_argument("--split", default="patient", help="Kept for compatibility; grouping is controlled by --group_by.")
    parser.add_argument("--group_by", nargs="+", default=["patient_id", "case_id", "series_uid", "source_series_dir"])
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    summary = build_splits(
        args.metadata_dir,
        args.output_dir,
        train=args.train,
        val=args.val,
        test=args.test,
        group_by=args.group_by,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
