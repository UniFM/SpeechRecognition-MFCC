from __future__ import annotations

import numpy as np
from scipy.signal import stft

####################### STEP 4 : STFT，即对分帧之后的每一帧做DFT（离散傅里叶变换） #######################
def compute_stft_per_frame(windowed_frames: np.ndarray, n_fft: int) -> np.ndarray:
    """对已分帧加窗后的每一帧做 STFT（避免重复加窗和重复分帧）。"""
    if windowed_frames.ndim != 2:
        raise ValueError("windowed_frames 必须是二维矩阵")
    frame_length = windowed_frames.shape[1]
    # spectra：先用 np.empty 创建一个空数组，用来存放所有帧的频谱结果，数据类型是复数（np.complex128），
    # 因为傅里叶变换的结果包含实部和虚部（或幅度和相位）
    # n_fft // 2 + 1：这是单边谱（One-sided Spectrum）的频点数量。
    # 根据奈奎斯特采样定理，对于实数信号，FFT 的结果是对称的，只需要取前一半（从 0 到 N_fft/2）
    spectra = np.empty((windowed_frames.shape[0], n_fft // 2 + 1), dtype=np.complex128)

    for idx, frame in enumerate(windowed_frames):
        # 调用的是 SciPy 里的底层 FFT 算法
        # frame：每一帧的信号
        # fs：采样率，这里设置为 1.0
        # window：窗口函数，这里设置为 "boxcar"，即矩形窗（不加窗），已经乘过汉明窗了！！
        # nperseg：每一帧的点数，即帧长
        # noverlap：每一帧的overlap，即帧移
        # nfft：FFT 点数
        # boundary：边界处理，这里设置为 None，即不处理
        # padded：是否填充，这里设置为 False，即不填充
        # stft 是 SciPy 里的底层 FFT 算法，返回值是：频率 bins，时间 bins，频谱
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
        # zxx 形状为 (freq_bins, 1)，取第一列
        spectra[idx] = zxx[:, 0]

    # scaling="spectrum" 时 SciPy 输出 = DFT / sum(window)；矩形窗之和为 frame_length，
    # 乘回 frame_length 即恢复未缩放的 DFT 标度 X[k] = sum_n x[n] e^{-j2πkn/N}
    return spectra * frame_length


def compute_power_spectrum(stft_matrix: np.ndarray) -> np.ndarray:
    """计算单边功率谱 P[k] = |X[k]|^2。"""
    if stft_matrix.ndim != 2:
        raise ValueError("stft_matrix 必须是二维矩阵")
    return np.abs(stft_matrix) ** 2
