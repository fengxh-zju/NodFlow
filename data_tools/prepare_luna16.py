import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.metrics.histogram import compute_histogram
from src.utils.geometry import sphere_mask
from src.utils.io import ensure_dir, save_json, save_volume


DTYPE_MAP = {
    "MET_CHAR": np.int8,
    "MET_UCHAR": np.uint8,
    "MET_SHORT": np.int16,
    "MET_USHORT": np.uint16,
    "MET_INT": np.int32,
    "MET_UINT": np.uint32,
    "MET_FLOAT": np.float32,
    "MET_DOUBLE": np.float64,
}


def parse_mhd(path):
    header = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
    return header


def read_mhd(path):
    path = Path(path)
    header = parse_mhd(path)
    dims = tuple(int(x) for x in header["DimSize"].split())
    dtype = DTYPE_MAP[header.get("ElementType", "MET_SHORT")]
    raw_path = path.parent / header["ElementDataFile"]
    arr = np.fromfile(raw_path, dtype=dtype)
    if header.get("BinaryDataByteOrderMSB", "False") == "True":
        arr = arr.byteswap().newbyteorder()
    arr = arr.reshape((dims[2], dims[1], dims[0]))
    arr = np.transpose(arr, (2, 1, 0)).astype(np.float32)
    spacing = np.array([float(x) for x in header.get("ElementSpacing", "1 1 1").split()], dtype=np.float32)
    origin = np.array([float(x) for x in header.get("Offset", header.get("Position", "0 0 0")).split()], dtype=np.float32)
    # MetaImage serializes the direction matrix by columns. Transpose the
    # header values so world = origin + direction @ (index * spacing), matching
    # SimpleITK's TransformPhysicalPointToContinuousIndex convention.
    transform = np.array(
        [float(x) for x in header.get("TransformMatrix", "1 0 0 0 1 0 0 0 1").split()],
        dtype=np.float32,
    ).reshape(3, 3).T
    return arr, spacing, origin, transform


def world_to_voxel(world, spacing, origin, transform):
    world = np.asarray(world, dtype=np.float32)
    return np.linalg.inv(transform).dot(world - origin) / spacing


def crop_with_pad(volume, center, size):
    size = np.asarray(size, dtype=int)
    center = np.asarray(center, dtype=int)
    start = center - size // 2
    end = start + size
    out = np.full(tuple(size), -1000.0, dtype=np.float32)
    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, volume.shape)
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    if np.all(src_end > src_start):
        out[dst_start[0]:dst_end[0], dst_start[1]:dst_end[1], dst_start[2]:dst_end[2]] = volume[
            src_start[0]:src_end[0], src_start[1]:src_end[1], src_start[2]:src_end[2]
        ]
    return out


def load_annotations(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def index_mhd(raw_root):
    index = {}
    for path in sorted(Path(raw_root).rglob("*.mhd")):
        header = parse_mhd(path)
        data_file = header.get("ElementDataFile", "")
        if not data_file.endswith(".raw"):
            continue
        if any(part.lower() == "seg-lungs-luna16" for part in path.parts):
            continue
        previous = index.get(path.stem)
        if previous is not None and previous.resolve() != path.resolve():
            raise ValueError(
                f"ambiguous non-segmentation MHD series {path.stem}: "
                f"{previous} and {path}"
            )
        index[path.stem] = path
    return index


def write_split(ids, output_root):
    split_dir = Path(output_root) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    n_train = max(1, int(0.7 * n)) if n else 0
    n_val = max(n_train + 1, int(0.8 * n)) if n > 2 else n
    splits = {
        "train": ids[:n_train],
        "val": ids[n_train:n_val],
        "test": ids[n_val:],
    }
    summary = {"num_groups": len(ids), "num_samples": len(ids), "splits": {}}
    for split, values in splits.items():
        text = "\n".join(values) + ("\n" if values else "")
        (split_dir / f"{split}_cases.txt").write_text(text, encoding="utf-8")
        (split_dir / f"{split}_groups.txt").write_text(text, encoding="utf-8")
        summary["splits"][split] = {"num_groups": len(values), "num_samples": len(values)}
    save_json(summary, split_dir / "split_summary.json")


def subtype_from_diameter(diameter_mm):
    if diameter_mm < 8:
        return "ground_glass"
    if diameter_mm < 15:
        return "part_solid"
    return "solid"


def prepare_luna16(raw_root, output_root, annotations_csv, roi_size, max_cases=None):
    raw_root = Path(raw_root)
    output_root = Path(output_root)
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

    annotations = load_annotations(annotations_csv)
    mhd_index = index_mhd(raw_root)
    written = []
    missing = []
    for idx, row in enumerate(annotations):
        if max_cases is not None and len(written) >= max_cases:
            break
        seriesuid = row["seriesuid"]
        mhd_path = mhd_index.get(seriesuid)
        if not mhd_path:
            missing.append(seriesuid)
            continue
        volume, spacing, origin, transform = read_mhd(mhd_path)
        center_world = np.array([float(row["coordX"]), float(row["coordY"]), float(row["coordZ"])], dtype=np.float32)
        diameter_mm = float(row["diameter_mm"])
        center_voxel = world_to_voxel(center_world, spacing, origin, transform)
        crop = crop_with_pad(volume, np.round(center_voxel).astype(int), roi_size)
        radius_vox = max(2.0, diameter_mm / float(np.mean(spacing)) / 2.0)
        mask = sphere_mask(tuple(roi_size), np.asarray(roi_size) // 2, radius_vox)
        subtype = subtype_from_diameter(diameter_mm)
        sample_id = f"luna16_{idx:06d}"
        save_volume(crop, output_root / "rois/pathological/images" / f"{sample_id}.nii.gz")
        save_volume(mask, output_root / "rois/pathological/masks" / f"{sample_id}.nii.gz")
        normal = crop.copy()
        normal[mask > 0] = np.median(crop[mask == 0]) if np.any(mask == 0) else -800.0
        save_volume(normal, output_root / "rois/normal/images" / f"{sample_id}.nii.gz")
        save_volume(np.zeros_like(mask, dtype=np.float32), output_root / "rois/normal/masks" / f"{sample_id}.nii.gz")
        meta = {
            "sample_id": sample_id,
            "case_id": seriesuid,
            "nodule_id": f"n{idx:06d}",
            "center_world": center_world.tolist(),
            "center_voxel": [float(x) for x in center_voxel],
            "diameter_mm": diameter_mm,
            "subtype": subtype,
            "histogram": compute_histogram(crop[mask > 0]).tolist(),
            "source_dataset": "LUNA16",
            "source_mhd": str(mhd_path),
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
            "num_written": len(written),
            "num_missing_series": len(set(missing)),
            "missing_series_sample": sorted(set(missing))[:20],
            "roi_size": list(roi_size),
        },
        output_root / "prepare_luna16_summary.json",
    )
    print(f"prepare_luna16: wrote {len(written)} ROI samples to {output_root}")
    if missing:
        print(f"prepare_luna16: missing {len(set(missing))} series in current raw_root (this is expected for partial subset downloads)")


def main():
    parser = argparse.ArgumentParser(description="Prepare LUNA16 .mhd/.raw data into the standard ROI layout.")
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--annotations_csv", default=None)
    parser.add_argument("--output_root", default="processed/LUNA16")
    parser.add_argument("--roi_size", nargs=3, type=int, default=[64, 64, 64])
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--spacing", nargs="*", default=None)
    parser.add_argument("--hu_clip", nargs="*", default=None)
    args = parser.parse_args()
    annotations_csv = Path(args.annotations_csv) if args.annotations_csv else Path(args.raw_root) / "annotations.csv"
    prepare_luna16(args.raw_root, args.output_root, annotations_csv, tuple(args.roi_size), args.max_cases)


if __name__ == "__main__":
    main()
