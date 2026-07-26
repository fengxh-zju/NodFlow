import csv
import json
import os
import uuid
from pathlib import Path

import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _temporary_output_path(path):
    path = Path(path)
    temporary_root = path.parent / ".atomic_tmp"
    ensure_dir(temporary_root)
    return temporary_root / f"{uuid.uuid4().hex}-{path.name}"


def _cleanup_temporary(path):
    path = Path(path)
    path.unlink(missing_ok=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass


def save_json(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    payload = json.dumps(obj, indent=2, sort_keys=True)
    temporary = _temporary_output_path(path)
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        _cleanup_temporary(temporary)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_volume(array, path, spacing=(1.0, 1.0, 1.0)):
    path = Path(path)
    ensure_dir(path.parent)
    arr = np.asarray(array, dtype=np.float32)
    temporary = _temporary_output_path(path)
    try:
        saved_as_nifti = False
        try:
            import nibabel as nib

            affine = np.diag(
                [float(spacing[0]), float(spacing[1]), float(spacing[2]), 1.0]
            )
            nib.save(nib.Nifti1Image(arr, affine), str(temporary))
            saved_as_nifti = True
        except Exception:
            temporary.unlink(missing_ok=True)
        if not saved_as_nifti:
            with temporary.open("wb") as f:
                np.savez_compressed(
                    f, image=arr, spacing=np.asarray(spacing, dtype=np.float32)
                )
        os.replace(temporary, path)
    finally:
        _cleanup_temporary(temporary)


def load_volume(path):
    path = Path(path)
    try:
        import nibabel as nib

        return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    except Exception:
        payload = np.load(path, allow_pickle=False)
        if isinstance(payload, np.lib.npyio.NpzFile):
            return payload["image"].astype(np.float32)
        return np.asarray(payload, dtype=np.float32)


def list_files(root, suffix=".nii.gz"):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.name.endswith(suffix))


def write_jsonl(rows, path):
    path = Path(path)
    ensure_dir(path.parent)
    temporary = _temporary_output_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        _cleanup_temporary(temporary)


def write_csv_rows(rows, path, fieldnames=None):
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    temporary = _temporary_output_path(path)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        _cleanup_temporary(temporary)
