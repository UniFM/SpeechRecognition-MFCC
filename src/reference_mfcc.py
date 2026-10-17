"""librosa 参考对照模块：本项目中唯一允许调用现成 MFCC / Mel 高层函数的地方。"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.fft import dct

from src.preprocessing import build_hamming_window


def build_aligned_reference_input(
    signal: np.ndarray,
    frame_length: int,
    frame_step: int,
    n_fft: int,
    num_frames: int,
) -> np.ndarray:
    """构造与自定义分帧逐帧对齐的 librosa 输入信号。

    librosa 在 center=False 时：
    - 帧数 = 1 + (len(y) - n_fft) // hop，按 n_fft(512) 而非 win_length(400) 计算，
      且不对尾部补零，因此比自定义实现(尾部补零到整帧)少帧；
    - 400 点窗被居中补零到 512 点，即第 t 帧的窗覆盖 y[t*hop+56 : t*hop+456]。
    为此：左侧补 56 个零抵消窗的居中偏移，尾部补零到与自定义实现相同的填充长度，
    右侧再补 56 个零，使 librosa 恰好得到 num_frames 帧且每帧覆盖相同采样点。
    """
    padded_length = (num_frames - 1) * frame_step + frame_length
    if padded_length < signal.size:
        raise ValueError("num_frames 与信号长度不匹配")
    left = (n_fft - frame_length) // 2
    right = n_fft - frame_length - left
    return np.concatenate(
        [
            np.zeros(left),
            signal,
            np.zeros(padded_length - signal.size),
            np.zeros(right),
        ]
    )


def reference_config(
    sample_rate: int,
    frame_length: int,
    frame_step: int,
    n_fft: int,
    n_mels: int,
    n_mfcc: int,
    fmin: float,
    fmax: float,
) -> dict[str, object]:
    """librosa.feature.mfcc 的完整调用参数（除 y / window 数组外）。"""
    return {
        "sr": sample_rate,
        "n_mfcc": n_mfcc + 1,
        "n_fft": n_fft,
        "hop_length": frame_step,
        "win_length": frame_length,
        "center": False,
        "power": 2.0,
        "n_mels": n_mels,
        "fmin": fmin,
        "fmax": fmax,
        "htk": True,
        "mel_norm": None,
        "dct_type": 2,
        "norm": None,
        "lifter": 0,
    }


def compute_reference_mfcc(
    signal: np.ndarray,
    num_frames: int,
    sample_rate: int,
    frame_length: int,
    frame_step: int,
    n_fft: int,
    n_mels: int,
    n_mfcc: int,
    fmin: float,
    fmax: float,
) -> dict[str, np.ndarray]:
    """调用 librosa 计算参考结果，全部返回 (num_frames, dim) 形状。

    返回：
    - mfcc_librosa：librosa.feature.mfcc 原始输出的索引 1..n_mfcc（dB 标度）；
    - log_mel_librosa：librosa Mel 功率谱取自然对数，用于逐级定位差异；
    - mfcc_librosa_ln：对 log_mel_librosa 做相同 DCT 得到的自然对数标度 MFCC。
    """
    y = build_aligned_reference_input(signal, frame_length, frame_step, n_fft, num_frames)
    window = build_hamming_window(frame_length)
    cfg = reference_config(
        sample_rate, frame_length, frame_step, n_fft, n_mels, n_mfcc, fmin, fmax
    )

    mfcc_full = librosa.feature.mfcc(y=y, window=window, **cfg)

    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=frame_step,
        win_length=frame_length,
        window=window,
        center=False,
        power=2.0,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=True,
        norm=None,
    )
    log_mel_librosa = np.log(mel_power.T + 1e-10)
    mfcc_librosa_ln = dct(log_mel_librosa, type=2, axis=1, norm=None)[:, 1 : n_mfcc + 1]

    return {
        "mfcc_librosa": mfcc_full[1 : n_mfcc + 1, :].T,
        "log_mel_librosa": log_mel_librosa,
        "mfcc_librosa_ln": mfcc_librosa_ln,
    }


def error_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """逐元素比较，返回 MAE / RMSE / 最大绝对误差。形状不一致时直接报错，不截断。"""
    if a.shape != b.shape:
        raise ValueError(f"形状不一致，无法逐元素比较: {a.shape} vs {b.shape}")
    diff = a - b
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
    }
