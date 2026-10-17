from __future__ import annotations

import numpy as np
from scipy.signal import stft


def compute_stft_per_frame(windowed_frames: np.ndarray, n_fft: int) -> np.ndarray:
    """对已分帧加窗后的每一帧做 STFT（避免重复加窗和重复分帧）。"""
    if windowed_frames.ndim != 2:
        raise ValueError("windowed_frames 必须是二维矩阵")
    frame_length = windowed_frames.shape[1]
    spectra = np.empty((windowed_frames.shape[0], n_fft // 2 + 1), dtype=np.complex128)

    for idx, frame in enumerate(windowed_frames):
        _, _, zxx = stft(
            frame,
            fs=1.0,
            window="boxcar",
            nperseg=frame_length,
            noverlap=0,
            nfft=n_fft,
            boundary=None,
            padded=False,
            return_onesided=True,
            scaling="spectrum",
        )
        # zxx 形状为 (freq_bins, 1)
        if zxx.shape[1] != 1:
            raise ValueError(f"STFT 输出异常形状: {zxx.shape}")
        spectra[idx] = zxx[:, 0]

    # scaling="spectrum" 时 SciPy 输出 = DFT / sum(window)；矩形窗之和为 frame_length，
    # 乘回 frame_length 即恢复未缩放的 DFT 标度 X[k] = sum_n x[n] e^{-j2πkn/N}
    return spectra * frame_length


def compute_power_spectrum(stft_matrix: np.ndarray) -> np.ndarray:
    """计算单边功率谱 P[k] = |X[k]|^2。"""
    if stft_matrix.ndim != 2:
        raise ValueError("stft_matrix 必须是二维矩阵")
    return np.abs(stft_matrix) ** 2
