from __future__ import annotations

import numpy as np


def _edge_padded(mat: np.ndarray, pad: int) -> np.ndarray:
    if mat.ndim == 1:
        mat = mat[:, None]
    return np.pad(mat, ((pad, pad), (0, 0)), mode="edge")


def compute_delta(features: np.ndarray, window: int = 2) -> np.ndarray:
    """使用回归公式计算 Delta，按时间轴求导。"""
    if window < 1:
        raise ValueError("window 必须 >= 1")
    if features.ndim == 1:
        features = features[:, None]
    if features.ndim != 2:
        raise ValueError("features 必须为二维")

    padded = _edge_padded(features, window)
    denom = 2.0 * sum(r * r for r in range(1, window + 1))
    delta = np.zeros_like(features, dtype=np.float64)

    for t in range(features.shape[0]):
        acc = np.zeros(features.shape[1], dtype=np.float64)
        for r in range(1, window + 1):
            acc += r * (padded[t + window + r] - padded[t + window - r])
        delta[t] = acc / denom
    return delta


def compute_delta_delta(features: np.ndarray, window: int = 2) -> np.ndarray:
    """对 Delta 再求一次导数，得到 Delta-Delta。"""
    return compute_delta(compute_delta(features, window=window), window=window)
