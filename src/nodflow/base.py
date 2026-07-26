import numpy as np

from src.utils.config import load_config, load_method_config


class NodFlowRuntime:
    name = "nodflow"

    def __init__(self, method_config, data_config):
        self.data_config = load_config(data_config)
        self.method_config = load_method_config(method_config)

    def _preprocess_source(self, image):
        image = np.asarray(image, dtype=np.float32)
        hu_clip = (self.data_config.get("preprocess") or {}).get("hu_clip")
        if hu_clip is not None:
            image = np.clip(image, float(hu_clip[0]), float(hu_clip[1]))
        return image.astype(np.float32)
