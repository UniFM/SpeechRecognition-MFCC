####################### STEP 8 : 拼接特征，得到 39 维特征 #######################

from __future__ import annotations

import numpy as np


def assemble_static_feature(static_mfcc: np.ndarray, log_energy: np.ndarray) -> np.ndarray:
    """拼接 12 维静态 MFCC 与 1 维对数能量得到 13 维静态特征。"""
    if static_mfcc.ndim != 2 or log_energy.ndim != 2:
        raise ValueError("输入必须为二维矩阵")
    if static_mfcc.shape[0] != log_energy.shape[0]:
        raise ValueError("帧数不一致")
    # axis=1：沿着每一行（即跨越不同的梅尔频带）进行拼接
    return np.concatenate([static_mfcc, log_energy], axis=1)


def assemble_final_feature(
    static_feature: np.ndarray,
    delta_feature: np.ndarray,
    delta_delta_feature: np.ndarray,
) -> np.ndarray:
    """拼接静态、Delta、Delta-Delta，得到 39 维特征。"""
    if not (
        static_feature.shape == delta_feature.shape == delta_delta_feature.shape
    ):
        raise ValueError("静态、Delta、Delta-Delta 形状必须完全一致")
    return np.concatenate([static_feature, delta_feature, delta_delta_feature], axis=1)


def cmvn(feature: np.ndarray, epsilon: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按时间轴进行 CMVN，返回 (归一化特征, 均值, 标准差)。"""
    if feature.ndim != 2:
        raise ValueError("feature 必须为二维矩阵")
    mean = np.mean(feature, axis=0, keepdims=True)
    std = np.sqrt(np.mean((feature - mean) ** 2, axis=0, keepdims=True))
    normalized = (feature - mean) / (std + epsilon)
    return normalized, mean, std
