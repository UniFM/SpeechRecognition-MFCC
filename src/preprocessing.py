from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(
    audio_path: Path,
    expected_sample_rate: int = 16000,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[np.ndarray, int, int]:
    """读取 WAV 并进行单声道、采样率、数值合法性检查。

    返回 (signal, sample_rate, original_channels)。
    soundfile 对 16-bit PCM 默认归一化到 [-1, 1) 的 float64。
    """
    signal, sample_rate = sf.read(str(audio_path), dtype="float64", always_2d=True)
    original_channels = int(signal.shape[1])

    if signal.shape[0] == 0:
        raise ValueError(f"音频为空: {audio_path}")

    if original_channels == 1:
        signal = signal[:, 0]
    else:
        # 明确使用均值混音规则转换为单声道
        signal = signal.mean(axis=1)

    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"采样率不符合要求: {sample_rate} Hz, 期望 {expected_sample_rate} Hz, 文件: {audio_path}"
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(f"音频包含 NaN/Inf: {audio_path}")

    peak = float(np.max(np.abs(signal)))
    if peak > 1.0:
        # 不裁剪，仅提示：超出 [-1, 1] 说明输入并非常规归一化 PCM
        print(f"[INFO] 音频峰值 {peak:.4f} 超出 [-1, 1]，未做裁剪: {audio_path}")

    if start_time is not None or end_time is not None:
        start = 0.0 if start_time is None else float(start_time)
        end = len(signal) / sample_rate if end_time is None else float(end_time)
        if start < 0 or end <= start:
            raise ValueError(f"时间片段非法: start_time={start}, end_time={end}")
        if end > len(signal) / sample_rate:
            raise ValueError(
                f"end_time 超出音频时长: end_time={end}, duration={len(signal) / sample_rate}"
            )
        start_idx = int(round(start * sample_rate))
        end_idx = int(round(end * sample_rate))
        signal = signal[start_idx:end_idx]
        if signal.size == 0:
            raise ValueError("截取后音频为空")

    return signal.astype(np.float64), sample_rate, original_channels


def summarize_signal(signal: np.ndarray) -> dict[str, float]:
    """统计信号范围、均值和 RMS。"""
    rms = float(np.sqrt(np.mean(np.square(signal)))) if signal.size else 0.0
    return {
        "min": float(signal.min()) if signal.size else 0.0,
        "max": float(signal.max()) if signal.size else 0.0,
        "mean": float(signal.mean()) if signal.size else 0.0,
        "rms": rms,
    }


def pre_emphasis(signal: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """预加重：提升高频相对能量，缓解声道谱倾斜。"""
    if signal.ndim != 1:
        raise ValueError("pre_emphasis 输入必须是一维信号")
    emphasized = np.empty_like(signal, dtype=np.float64)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - alpha * signal[:-1]
    return emphasized


def frame_signal(
    signal: np.ndarray,
    frame_length: int,
    frame_step: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """按固定帧长和帧移分帧，尾部不足补零。"""
    if frame_length <= 0 or frame_step <= 0:
        raise ValueError("frame_length 和 frame_step 必须为正整数")
    if signal.ndim != 1:
        raise ValueError("frame_signal 输入必须是一维信号")
    if signal.size == 0:
        raise ValueError("无法对空信号分帧")

    num_samples = signal.size
    if num_samples <= frame_length:
        num_frames = 1
    else:
        num_frames = int(np.ceil((num_samples - frame_length) / frame_step)) + 1
    padded_length = (num_frames - 1) * frame_step + frame_length
    pad_width = padded_length - num_samples

    padded = np.pad(signal, (0, pad_width), mode="constant", constant_values=0.0)
    indices = (
        np.arange(num_frames)[:, None] * frame_step + np.arange(frame_length)[None, :]
    )
    frames = padded[indices]
    meta = {
        "num_samples": int(num_samples),
        "padded_length": int(padded_length),
        "pad_width": int(pad_width),
        "num_frames": int(num_frames),
        "frame_length": int(frame_length),
        "frame_step": int(frame_step),
    }
    return frames, meta


def build_hamming_window(frame_length: int) -> np.ndarray:
    """按定义手动构造 Hamming 窗。"""
    if frame_length <= 1:
        raise ValueError("frame_length 必须大于 1")
    n = np.arange(frame_length, dtype=np.float64)
    return 0.54 - 0.46 * np.cos((2.0 * np.pi * n) / (frame_length - 1))


def apply_window(frames: np.ndarray, window: np.ndarray) -> np.ndarray:
    """逐帧乘窗，保持矩阵形状不变。"""
    if frames.ndim != 2:
        raise ValueError("frames 必须是二维矩阵")
    if window.ndim != 1 or window.shape[0] != frames.shape[1]:
        raise ValueError("window 长度必须等于每帧采样点数")
    return frames * window[None, :]


def compute_log_energy(windowed_frames: np.ndarray, log_epsilon: float) -> np.ndarray:
    """根据加窗帧计算逐帧对数短时能量。"""
    if windowed_frames.ndim != 2:
        raise ValueError("windowed_frames 必须是二维矩阵")
    energy = np.sum(windowed_frames * windowed_frames, axis=1, keepdims=True)
    return np.log(energy + log_epsilon)
