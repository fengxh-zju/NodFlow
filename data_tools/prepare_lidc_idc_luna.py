import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from data_tools.prepare_luna16 import crop_with_pad, subtype_from_diameter, write_split
from src.metrics.histogram import compute_histogram
from src.utils.geometry import sphere_mask
from src.utils.io import ensure_dir, save_json, save_volume


STANDARD_DIRS = [
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
]


def standard_dirs(output_root):
    for rel in STANDARD_DIRS:
        ensure_dir(output_root / rel)


def load_annotations(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def index_ct_series(raw_root):
    index = {}
    for path in Path(raw_root).rglob("CT_*"):
        if not path.is_dir():
            continue
        series_uid = path.name[3:]
        if any(path.glob("*.dcm")):
            index.setdefault(series_uid, path)
    return index


def _slice_sort_key(ds, normal):
    image_position = np.asarray([float(x) for x in ds.ImagePositionPatient], dtype=np.float32)
    return float(np.dot(image_position, normal))


def read_dicom_ct_series(series_dir):
    import pydicom

    datasets = []
    for dicom_path in sorted(Path(series_dir).glob("*.dcm")):
        ds = pydicom.dcmread(str(dicom_path), force=True)
        if getattr(ds, "Modality", "CT") != "CT":
            continue
        if not hasattr(ds, "PixelData"):
            continue
        datasets.append(ds)
    if not datasets:
        raise ValueError(f"no CT slices with PixelData under {series_dir}")

    first = datasets[0]
    orientation = np.asarray([float(x) for x in first.ImageOrientationPatient], dtype=np.float32)
    row_cos = orientation[:3]
    col_cos = orientation[3:]
    normal = np.cross(row_cos, col_cos)
    datasets.sort(key=lambda ds: _slice_sort_key(ds, normal))

    pixel_spacing = [float(x) for x in first.PixelSpacing]
    if len(datasets) > 1:
        slice_positions = np.asarray([_slice_sort_key(ds, normal) for ds in datasets], dtype=np.float32)
        slice_spacing = float(np.median(np.diff(slice_positions)))
    else:
        slice_spacing = float(getattr(first, "SliceThickness", 1.0))
    slice_spacing = abs(slice_spacing) if slice_spacing else 1.0

    slices = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(arr * slope + intercept)

    stack = np.stack(slices, axis=0)
    volume = np.transpose(stack, (2, 1, 0)).astype(np.float32)
    spacing = np.asarray([pixel_spacing[1], pixel_spacing[0], slice_spacing], dtype=np.float32)
    origin = np.asarray([float(x) for x in datasets[0].ImagePositionPatient], dtype=np.float32)
    transform = np.stack([row_cos, col_cos, normal], axis=1).astype(np.float32)
    return volume, spacing, origin, transform, datasets


def world_to_voxel(world, spacing, origin, transform):
    world = np.asarray(world, dtype=np.float32)
    return np.linalg.inv(transform).dot(world - origin) / spacing


def normal_crop_from_pathological(crop, mask):
    normal = crop.copy()
    fill_value = np.median(crop[mask == 0]) if np.any(mask == 0) else -800.0
    normal[mask > 0] = fill_value
    return normal


def prepare_lidc_idc_luna(raw_root, annotations_csv, output_root, roi_size, max_samples=None):
    raw_root = Path(raw_root)
    annotations_csv = Path(annotations_csv)
    output_root = Path(output_root)
    standard_dirs(output_root)

    ct_index = index_ct_series(raw_root)
    annotations = load_annotations(annotations_csv)
    grouped = defaultdict(list)
    for idx, row in enumerate(annotations):
        grouped[row["seriesuid"]].append((idx, row))

    written = []
    missing_series = []
    failed_series = []
    failed_rows = []

    for series_uid in sorted(grouped):
        if max_samples is not None and len(written) >= max_samples:
            break
        series_dir = ct_index.get(series_uid)
        if not series_dir:
            missing_series.append(series_uid)
            continue
        try:
            volume, spacing, origin, transform, datasets = read_dicom_ct_series(series_dir)
        except Exception as exc:
            failed_series.append({"seriesuid": series_uid, "reason": str(exc), "series_dir": str(series_dir)})
            continue

        for annotation_idx, row in grouped[series_uid]:
            if max_samples is not None and len(written) >= max_samples:
                break
            try:
                center_world = np.asarray(
                    [float(row["coordX"]), float(row["coordY"]), float(row["coordZ"])],
                    dtype=np.float32,
                )
                diameter_mm = float(row["diameter_mm"])
                center_voxel = world_to_voxel(center_world, spacing, origin, transform)
                crop = crop_with_pad(volume, np.round(center_voxel).astype(int), roi_size)
            except Exception as exc:
                failed_rows.append({"row": annotation_idx, "seriesuid": series_uid, "reason": str(exc)})
                continue

            radius_vox = max(2.0, diameter_mm / float(np.mean(spacing)) / 2.0)
            mask = sphere_mask(tuple(roi_size), np.asarray(roi_size) // 2, radius_vox)
            sample_id = f"lidc_idc_luna_{annotation_idx:06d}"
            subtype = subtype_from_diameter(diameter_mm)

            save_volume(crop, output_root / "rois/pathological/images" / f"{sample_id}.nii.gz")
            save_volume(mask, output_root / "rois/pathological/masks" / f"{sample_id}.nii.gz")
            normal = normal_crop_from_pathological(crop, mask)
            save_volume(normal, output_root / "rois/normal/images" / f"{sample_id}.nii.gz")
            save_volume(np.zeros_like(mask, dtype=np.float32), output_root / "rois/normal/masks" / f"{sample_id}.nii.gz")

            meta = {
                "sample_id": sample_id,
                "case_id": series_uid,
                "nodule_id": f"luna_annotation_{annotation_idx:06d}",
                "center_world": center_world.tolist(),
                "center_voxel": [float(x) for x in center_voxel],
                "diameter_mm": diameter_mm,
                "subtype": subtype,
                "histogram": compute_histogram(crop[mask > 0]).tolist(),
                "source_dataset": "LIDC-IDRI-IDC-with-LUNA16-annotations",
                "source_series_dir": str(series_dir),
                "num_slices": len(datasets),
                "spacing": [float(x) for x in spacing],
                "origin": [float(x) for x in origin],
            }
            save_json(meta, output_root / "metadata" / f"{sample_id}.json")
            save_json(meta, output_root / "rois/pathological/metadata" / f"{sample_id}.json")
            save_json({**meta, "subtype": "normal"}, output_root / "rois/normal/metadata" / f"{sample_id}.json")
            written.append(sample_id)

    write_split(written, output_root)
    save_json(
        {
            "raw_root": str(raw_root),
            "annotations_csv": str(annotations_csv),
            "num_annotations": len(annotations),
            "num_annotation_series": len(grouped),
            "num_ct_series_indexed": len(ct_index),
            "num_written": len(written),
            "num_missing_series": len(set(missing_series)),
            "missing_series_sample": sorted(set(missing_series))[:20],
            "num_failed_series": len(failed_series),
            "failed_series_sample": failed_series[:20],
            "num_failed_rows": len(failed_rows),
            "failed_rows_sample": failed_rows[:20],
            "roi_size": list(roi_size),
        },
        output_root / "prepare_lidc_idc_luna_summary.json",
    )
    print(f"prepare_lidc_idc_luna: wrote {len(written)} ROI samples to {output_root}")
    if missing_series:
        print(f"prepare_lidc_idc_luna: missing {len(set(missing_series))} annotation series in current raw_root")
    if failed_series or failed_rows:
        print(
            "prepare_lidc_idc_luna: skipped "
            f"{len(failed_series)} series and {len(failed_rows)} rows with read/geometry errors"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare IDC LIDC-IDRI CT DICOM using LUNA16 annotations into the standard ROI layout."
    )
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--roi_size", nargs=3, type=int, default=[64, 64, 64])
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    prepare_lidc_idc_luna(
        args.raw_root,
        args.annotations_csv,
        Path(args.output_root),
        tuple(args.roi_size),
        args.max_samples,
    )


if __name__ == "__main__":
    main()
