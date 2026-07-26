import argparse
import shutil
from pathlib import Path

import numpy as np

from src.metrics.histogram import compute_histogram
from src.utils.io import ensure_dir, load_json, load_volume, save_json, save_volume


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


def _read_ids(path):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--split_cases", default=None, help="Optional train_cases.txt-style filter.")
    args = parser.parse_args()
    roi_root = Path(args.roi_root)
    out = Path(args.output_root)
    for rel in ["masks", "histograms", "metadata"]:
        folder = out / rel
        if folder.exists():
            shutil.rmtree(folder)
        ensure_dir(folder)
    image_dir = roi_root / "images"
    mask_dir = roi_root / "masks"
    meta_dir = roi_root / "metadata"
    allowed_ids = _read_ids(args.split_cases)
    count = 0
    for image_path in sorted(image_dir.glob("*.nii.gz")):
        sid = image_path.name.replace(".nii.gz", "")
        if allowed_ids is not None and sid not in allowed_ids:
            continue
        mask = load_volume(mask_dir / image_path.name)
        image = load_volume(image_path)
        hist = compute_histogram(image[mask > 0])
        tid = f"target_{count:06d}"
        save_volume(mask, out / "masks" / f"{tid}.nii.gz")
        np.save(out / "histograms" / f"{tid}.npy", hist)
        meta = load_json(meta_dir / f"{sid}.json") if (meta_dir / f"{sid}.json").exists() else {}
        target_meta = {
            "target_id": tid,
            "subtype": meta.get("subtype", "unknown"),
            "diameter_mm": meta.get("diameter_mm"),
            "mask_path": f"masks/{tid}.nii.gz",
            "hist_path": f"histograms/{tid}.npy",
            "source_dataset": meta.get("source_dataset", "unknown"),
            "source_sample_id": sid,
            "case_id": meta.get("case_id"),
            "nodule_id": meta.get("nodule_id"),
        }
        target_meta.update(
            {key: meta[key] for key in SEMANTIC_FIELDS if meta.get(key) is not None}
        )
        save_json(
            target_meta,
            out / "metadata" / f"{tid}.json",
        )
        count += 1
    print(f"built {count} targets under {out}")


if __name__ == "__main__":
    main()
