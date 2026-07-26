import numpy as np


def sphere_mask(shape, center, radius):
    grid = np.indices(shape, dtype=np.float32)
    dist2 = sum((grid[i] - float(center[i])) ** 2 for i in range(3))
    return (dist2 <= float(radius) ** 2).astype(np.float32)


def boundary_band(mask):
    mask = mask > 0
    try:
        from scipy.ndimage import binary_dilation, binary_erosion

        return np.logical_xor(binary_dilation(mask, iterations=2), binary_erosion(mask, iterations=1))
    except Exception:
        shifted = np.zeros_like(mask, dtype=bool)
        for axis in range(3):
            shifted |= np.roll(mask, 1, axis=axis) ^ mask
            shifted |= np.roll(mask, -1, axis=axis) ^ mask
        return shifted
