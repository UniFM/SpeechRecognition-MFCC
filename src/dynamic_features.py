####################### STEP 7 : 计算 Delta 和 Delta-Delta #######################
# 静态 MFCC 只代表了某一帧的静态频谱，
# 而差分特征（Delta 及其加速度）能够反映语音信号随时间变化的动态趋势和斜率

from __future__ import annotations

import numpy as np


def _edge_padded(mat: np.ndarray, pad: int) -> np.ndarray:
    """在矩阵的上下两端（时间轴的头尾）进行边界填充"""
    if mat.ndim == 1:
        mat = mat[:, None]
    # 第一个元组 (pad, pad)：在行方向（时间轴）的头部和尾部各填充 pad 行
    # 第二个元组 (0, 0)：在列方向（梅尔频带）的头部和尾部各填充 0 列
    # mode="edge"：使用边缘填充模式，即用最外层的值填充
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
        # 创建一个用来累加特征维度的临时向量，它的长度等于特征的维数（即 features.shape[1]，比如 12 维）
        acc = np.zeros(features.shape[1], dtype=np.float64)
        for r in range(1, window + 1):
            acc += r * (padded[t + window + r] - padded[t + window - r])
        delta[t] = acc / denom
    return delta


def compute_delta_delta(features: np.ndarray, window: int = 2) -> np.ndarray:
    """对 Delta 再求一次导数，得到 Delta-Delta。"""
    return compute_delta(compute_delta(features, window=window), window=window)
