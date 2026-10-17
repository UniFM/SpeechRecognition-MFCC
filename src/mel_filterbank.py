from __future__ import annotations

import numpy as np


def hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray | float:
    """Hz -> Mel（课件公式）。"""
    return 1127.0 * np.log(1.0 + np.asarray(freq_hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Mel -> Hz（课件反公式）。"""
    return 700.0 * (np.exp(np.asarray(mel) / 1127.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """手动构建 Mel 三角滤波器组，返回 (filterbank, mel_points_hz)。"""
    if fmax > sample_rate / 2:
        raise ValueError("fmax 不能超过 Nyquist 频率")
    if fmin < 0 or fmax <= fmin:
        raise ValueError("fmin/fmax 参数非法")

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    freq_bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    max_bin = n_fft // 2
    freq_bins = np.clip(freq_bins, 0, max_bin)

    filterbank = np.zeros((n_mels, max_bin + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left = freq_bins[m - 1]
        center = freq_bins[m]
        right = freq_bins[m + 1]

        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        right = min(right, max_bin)
        center = min(center, right - 1)

        for k in range(left, center):
            denom = center - left
            if denom > 0:
                filterbank[m - 1, k] = (k - left) / denom
        filterbank[m - 1, center] = 1.0
        for k in range(center + 1, right + 1):
            denom = right - center
            if denom > 0:
                filterbank[m - 1, k] = (right - k) / denom

    return filterbank, hz_points


def apply_mel_filterbank(
    power_spectrum: np.ndarray,
    mel_filterbank: np.ndarray,
) -> np.ndarray:
    """功率谱乘 Mel 滤波器组，得到每帧 Mel 能量。"""
    if power_spectrum.ndim != 2:
        raise ValueError("power_spectrum 必须是二维矩阵")
    if mel_filterbank.ndim != 2:
        raise ValueError("mel_filterbank 必须是二维矩阵")
    if power_spectrum.shape[1] != mel_filterbank.shape[1]:
        raise ValueError("功率谱频率维度与 Mel 滤波器不一致")
    mel_energy = power_spectrum @ mel_filterbank.T
    return mel_energy
