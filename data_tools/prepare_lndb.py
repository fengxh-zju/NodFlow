import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_tools.prepare_luna16 import crop_with_pad, read_mhd, world_to_voxel, write_split
from src.metrics.histogram import compute_histogram
from src.utils.geometry import sphere_mask
from src.utils.io import ensure_dir, save_json, save_volume


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_lndb_id(value):
    return int(float(str(value).strip()))


def parse_float(row, key, default=None):
    value = row.get(key, "")
    if value in {"", None}:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def index_lndb_mhd(raw_root):
    result = {}
    for path in Path(raw_root).rglob("*.mhd"):
        stem = path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if not digits:
            continue
        result.setdefault(int(digits), path)
    return result


def subtype_from_lndb(row, diameter_mm):
    texture = parse_float(row, "Texture")
    if texture is not None:
        if texture <= 2:
            return "ground_glass"
        if texture <= 4:
            return "part_solid"
        return "solid"
    if diameter_mm < 8:
        return "ground_glass"
    if diameter_mm < 15:
        return "part_solid"
    return "solid"


def standard_dirs(output_root):
    for rel in [
        "imagesTr",
        "lung_masks",
        "nodule_masks",
        "metadata",
        "rois/pathological/images",
        "rois/pathological/masks",
        "rois/pathological/metadata",
        "rois/normal/images",
        "rois/normal/masks",
        "rois/normal/metadata",
        "splits",
    ]:
        ensure_dir(output_root / rel)


def prepare_lndb(raw_root, output_root, annotations_csv, roi_size, max_cases=None):
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    standard_dirs(output_root)

    rows = load_rows(annotations_csv)
    mhd_index = index_lndb_mhd(raw_root)
    written = []
    missing = []
    failed = []
    cached_mhd_path = None
    cached_mhd = None

    for idx, row in enumerate(rows):
        if max_cases is not None and len(written) >= max_cases:
            break
        try:
            lndb_id = parse_lndb_id(row["LNDbID"])
        except (KeyError, ValueError):
            failed.append({"row": idx, "reason": "invalid LNDbID"})
            continue
        mhd_path = mhd_index.get(lndb_id)
        if not mhd_path:
            missing.append(lndb_id)
            continue

        center_world = np.array(
            [parse_float(row, "x"), parse_float(row, "y"), parse_float(row, "z")],
            dtype=np.float32,
        )
        diameter_mm = parse_float(row, "DiamEq_Rad", default=6.0)
        if not np.all(np.isfinite(center_world)) or diameter_mm is None:
            failed.append({"row": idx, "LNDbID": lndb_id, "reason": "invalid coordinates or diameter"})
            continue

        try:
            if cached_mhd_path != mhd_path:
                cached_mhd = read_mhd(mhd_path)
                cached_mhd_path = mhd_path
            volume, spacing, origin, transform = cached_mhd
            center_voxel = world_to_voxel(center_world, spacing, origin, transform)
            if np.any(center_voxel < 0) or np.any(center_voxel >= np.asarray(volume.shape)):
                raise ValueError(
                    f"annotation center is outside the source volume: "
                    f"center={center_voxel.tolist()}, shape={list(volume.shape)}"
                )
            crop = crop_with_pad(volume, np.round(center_voxel).astype(int), roi_size)
        except Exception as exc:
            failed.append({"row": idx, "LNDbID": lndb_id, "reason": str(exc)})
            continue

        radius_vox = max(2.0, float(diameter_mm) / float(np.mean(spacing)) / 2.0)
        mask = sphere_mask(tuple(roi_size), np.asarray(roi_size) // 2, radius_vox)
        lesion = crop[mask > 0]
        if (
            lesion.size == 0
            or not np.all(np.isfinite(lesion))
            or float(np.max(lesion)) <= -990.0
        ):
            failed.append(
                {
                    "row": idx,
                    "LNDbID": lndb_id,
                    "reason": "invalid all-air or nonfinite annotated lesion crop",
                }
            )
            continue
        sample_id = f"lndb_{idx:06d}"
        subtype = subtype_from_lndb(row, float(diameter_mm))

        save_volume(crop, output_root / "rois/pathological/images" / f"{sample_id}.nii.gz")
        save_volume(mask, output_root / "rois/pathological/masks" / f"{sample_id}.nii.gz")
        normal = crop.copy()
        normal[mask > 0] = np.median(crop[mask == 0]) if np.any(mask == 0) else -800.0
        save_volume(normal, output_root / "rois/normal/images" / f"{sample_id}.nii.gz")
        save_volume(np.zeros_like(mask, dtype=np.float32), output_root / "rois/normal/masks" / f"{sample_id}.nii.gz")

        meta = {
            "sample_id": sample_id,
            "case_id": f"LNDb-{lndb_id:04d}",
            "nodule_id": str(row.get("FindingID", idx)),
            "center_world": center_world.tolist(),
            "center_voxel": [float(x) for x in center_voxel],
            "diameter_mm": float(diameter_mm),
            "subtype": subtype,
            "histogram": compute_histogram(lesion).tolist(),
            "source_dataset": "LNDb",
            "source_mhd": str(mhd_path),
            "spacing": [float(x) for x in spacing],
        }
        for key in [
            "Texture",
            "Calcification",
            "Malignancy",
            "Margin",
            "Sphericity",
            "Spiculation",
            "Subtlety",
            "Lobulation",
        ]:
            value = parse_float(row, key)
            if value is not None:
                meta[key.lower()] = value
        save_json(meta, output_root / "metadata" / f"{sample_id}.json")
        save_json(meta, output_root / "rois/pathological/metadata" / f"{sample_id}.json")
        save_json({**meta, "subtype": "normal"}, output_root / "rois/normal/metadata" / f"{sample_id}.json")
        written.append(sample_id)

    write_split(written, output_root)
    save_json(
        {
            "raw_root": str(raw_root),
            "annotations_csv": str(annotations_csv),
            "num_annotations": len(rows),
            "num_written": len(written),
            "num_missing_cases": len(set(missing)),
            "missing_cases_sample": [f"LNDb-{case_id:04d}" for case_id in sorted(set(missing))[:20]],
            "num_failed_rows": len(failed),
            "failed_rows_sample": failed[:20],
            "roi_size": list(roi_size),
        },
        output_root / "prepare_lndb_summary.json",
    )
    print(f"prepare_lndb: wrote {len(written)} ROI samples to {output_root}")
    if missing:
        print(f"prepare_lndb: missing {len(set(missing))} cases in current raw_root")
    if failed:
        print(f"prepare_lndb: skipped {len(failed)} rows with parsing/read errors")


def main():
    parser = argparse.ArgumentParser(description="Prepare LNDb MetaImage data into the standard ROI layout.")
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--annotations_csv", default=None)
    parser.add_argument("--output_root", default="processed/LNDb")
    parser.add_argument("--roi_size", nargs=3, type=int, default=[64, 64, 64])
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--spacing", nargs="*", default=None)
    parser.add_argument("--hu_clip", nargs="*", default=None)
    args = parser.parse_args()
    annotations_csv = Path(args.annotations_csv) if args.annotations_csv else Path(args.raw_root) / "allNods.csv"
    prepare_lndb(args.raw_root, args.output_root, annotations_csv, tuple(args.roi_size), args.max_cases)


if __name__ == "__main__":
    main()
