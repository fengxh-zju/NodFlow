from pathlib import Path

from src.utils.config import load_config
from src.utils.io import list_files, load_json, load_volume


def split_dir_for(config, root):
    return Path(config.get("splits") or Path(root) / "splits")


def read_split_ids(config, root, split):
    split_path = split_dir_for(config, root) / f"{split}_cases.txt"
    if not split_path.exists():
        return None
    return {line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def list_roi_image_paths(data_config, kind="normal", split=None):
    config = load_config(data_config) if isinstance(data_config, (str, Path)) else data_config
    root = Path(config["root"])
    image_dir = root / "rois" / kind / "images"
    paths = list_files(image_dir)
    if split is None:
        return paths
    split_ids = read_split_ids(config, root, split)
    if split_ids is None:
        return paths
    return [path for path in paths if path.name.replace(".nii.gz", "") in split_ids]


class ROIDataset:
    def __init__(self, data_config, split="train", kind="pathological"):
        self.config = load_config(data_config) if isinstance(data_config, (str, Path)) else data_config
        self.root = Path(self.config["root"])
        self.split = split
        self.kind = kind
        self.image_dir = self.root / "rois" / kind / "images"
        self.mask_dir = self.root / "rois" / kind / "masks"
        self.meta_dir = self.root / "rois" / kind / "metadata"
        ids = [p.name.replace(".nii.gz", "") for p in list_files(self.image_dir)]
        split_ids = read_split_ids(self.config, self.root, split)
        self.ids = [sid for sid in ids if split_ids is None or sid in split_ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        meta_path = self.meta_dir / f"{sid}.json"
        return {
            "sample_id": sid,
            "image": load_volume(self.image_dir / f"{sid}.nii.gz"),
            "mask": load_volume(self.mask_dir / f"{sid}.nii.gz"),
            "metadata": load_json(meta_path) if meta_path.exists() else {},
        }
