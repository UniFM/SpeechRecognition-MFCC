####################### STEP 6 : 取对数 + DCT，提取 MFCC 特征 #######################
"""目的：去相关"""
"""原因：
临近的梅尔滤波器之间有很大的重叠区域（三角形互相交叉），这意味着相邻的梅尔能量数据之间存在很高的冗余和强相关性
即 MFCC 特征的各个维度之间存在较强的相关性，这会影响到特征的区分能力和鲁棒性。
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct


def compute_log_mel(mel_energy: np.ndarray, epsilon: float) -> np.ndarray:
    """对 Mel 能量取自然对数并做数值保护。"""
    if mel_energy.ndim != 2:
        raise ValueError("mel_energy 必须是二维矩阵")
    return np.log(mel_energy + epsilon)


def compute_dct(log_mel: np.ndarray) -> np.ndarray:
    """DCT-II, norm=None，与课件公式一致。"""
    if log_mel.ndim != 2:
        raise ValueError("log_mel 必须是二维矩阵")
    # axis=1：沿着每一行（即跨越不同的梅尔频带）进行 DCT 变换
    # norm=None：不进行归一化，与课件公式一致
    return dct(log_mel, type=2, axis=1, norm=None)


def extract_static_mfcc(full_dct: np.ndarray, n_mfcc: int) -> np.ndarray:
    """保留索引 1..n_mfcc，丢弃 c0。"""
    if full_dct.ndim != 2:
        raise ValueError("full_dct 必须是二维矩阵")
    if full_dct.shape[1] < n_mfcc + 1:
        raise ValueError("DCT 维度不足，无法提取指定数量的 MFCC")
    return full_dct[:, 1 : n_mfcc + 1]
