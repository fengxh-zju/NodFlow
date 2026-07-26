import numpy as np


def compute_histogram(values, bins=40, hu_range=(-1000, 400), normalize=True, eps=1e-8):
    hist, _ = np.histogram(np.asarray(values).ravel(), bins=bins, range=hu_range)
    hist = hist.astype(np.float64) + eps
    if normalize:
        hist = hist / hist.sum()
    return hist.astype(np.float32)


def js_divergence(p, q, eps=1e-8):
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
