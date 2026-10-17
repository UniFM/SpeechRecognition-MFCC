from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import MFCCConfig
from src.dynamic_features import compute_delta, compute_delta_delta
from src.feature_transform import assemble_final_feature, assemble_static_feature, cmvn
from src.mel_filterbank import apply_mel_filterbank, build_mel_filterbank
from src.mfcc import compute_dct, compute_log_mel, extract_static_mfcc
from src.preprocessing import (
    apply_window,
    build_hamming_window,
    compute_log_energy,
    frame_signal,
    load_audio,
    pre_emphasis,
    summarize_signal,
)
from src.stft import compute_power_spectrum, compute_stft_per_frame


@dataclass(slots=True)
class PipelineResult:
    """保存单文件处理结果与中间数组。"""

    audio_path: Path
    sample_rate: int
    duration_sec: float
    num_samples: int
    status: str
    error: str
    arrays: dict[str, np.ndarray]
    metrics: dict[str, float | int]


def run_mfcc_pipeline(audio_path: Path, cfg: MFCCConfig) -> PipelineResult:
    """执行完整手写 MFCC 39 维特征流水线。"""
    arrays: dict[str, np.ndarray] = {}
    metrics: dict[str, float | int] = {}
    try:
        signal, sr, channels = load_audio(
            audio_path=audio_path,
            expected_sample_rate=cfg.sample_rate,
            start_time=cfg.start_time,
            end_time=cfg.end_time,
        )
        arrays["raw_signal"] = signal
        metrics["channels"] = channels
        metrics.update(summarize_signal(signal))

        emphasized = pre_emphasis(signal, alpha=cfg.pre_emphasis_alpha)
        arrays["pre_emphasis_signal"] = emphasized
        metrics.update({f"pre_{k}": v for k, v in summarize_signal(emphasized).items()})

        frames, frame_meta = frame_signal(
            emphasized,
            frame_length=cfg.frame_length,
            frame_step=cfg.frame_step,
        )
        arrays["frames"] = frames
        metrics.update(frame_meta)

        hamming = build_hamming_window(cfg.frame_length)
        arrays["hamming_window"] = hamming
        windowed_frames = apply_window(frames, hamming)
        arrays["windowed_frames"] = windowed_frames

        stft_matrix = compute_stft_per_frame(windowed_frames, n_fft=cfg.n_fft)
        arrays["stft_matrix"] = stft_matrix
        magnitude = np.abs(stft_matrix)
        arrays["magnitude_spectrum"] = magnitude
        power_spectrum = compute_power_spectrum(stft_matrix)
        arrays["power_spectrum"] = power_spectrum

        mel_filterbank, mel_hz_points = build_mel_filterbank(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            n_mels=cfg.n_mels,
            fmin=cfg.mel_fmin,
            fmax=cfg.mel_fmax,
        )
        arrays["mel_filterbank"] = mel_filterbank
        arrays["mel_hz_points"] = mel_hz_points

        mel_energy = apply_mel_filterbank(power_spectrum, mel_filterbank)
        arrays["mel_energy"] = mel_energy
        log_mel = compute_log_mel(mel_energy, epsilon=cfg.log_epsilon)
        arrays["log_mel"] = log_mel
        dct_full = compute_dct(log_mel)
        arrays["dct_full"] = dct_full
        static_mfcc = extract_static_mfcc(dct_full, n_mfcc=cfg.n_mfcc)
        arrays["static_mfcc"] = static_mfcc

        log_energy = compute_log_energy(windowed_frames, log_epsilon=cfg.log_epsilon)
        arrays["log_energy"] = log_energy

        static_feature = assemble_static_feature(static_mfcc, log_energy)
        arrays["static_feature"] = static_feature
        delta = compute_delta(static_feature, window=cfg.delta_window)
        arrays["delta_feature"] = delta
        delta_delta = compute_delta_delta(static_feature, window=cfg.delta_window)
        arrays["delta_delta_feature"] = delta_delta

        feature_39 = assemble_final_feature(static_feature, delta, delta_delta)
        arrays["feature_39_before_cmvn"] = feature_39
        feature_39_cmvn, mean, std = cmvn(feature_39, epsilon=cfg.cmvn_epsilon)
        arrays["feature_39_after_cmvn"] = feature_39_cmvn
        arrays["cmvn_mean"] = mean
        arrays["cmvn_std"] = std

        _assert_shapes(arrays, cfg.n_mels)
        _assert_finite(arrays)

        duration = signal.size / sr
        metrics.update(
            {
                "sample_rate": sr,
                "duration_sec": duration,
                "num_samples": int(signal.size),
                "num_frames": int(frames.shape[0]),
                "mfcc_rows": int(static_mfcc.shape[0]),
                "mfcc_cols": int(static_mfcc.shape[1]),
                "final_rows": int(feature_39_cmvn.shape[0]),
                "final_cols": int(feature_39_cmvn.shape[1]),
            }
        )
        return PipelineResult(
            audio_path=audio_path,
            sample_rate=sr,
            duration_sec=duration,
            num_samples=signal.size,
            status="ok",
            error="",
            arrays=arrays,
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        return PipelineResult(
            audio_path=audio_path,
            sample_rate=0,
            duration_sec=0.0,
            num_samples=0,
            status="failed",
            error=str(exc),
            arrays=arrays,
            metrics=metrics,
        )


def _assert_shapes(arrays: dict[str, np.ndarray], n_mels: int) -> None:
    if arrays["frames"].shape[1] != 400:
        raise ValueError(f"分帧长度应为 400，实际 {arrays['frames'].shape[1]}")
    if arrays["windowed_frames"].shape != arrays["frames"].shape:
        raise ValueError("加窗后形状应与分帧一致")
    if arrays["power_spectrum"].shape[1] != 257:
        raise ValueError(f"单边功率谱频率 bin 应为 257，实际 {arrays['power_spectrum'].shape[1]}")
    if arrays["mel_filterbank"].shape != (n_mels, 257):
        raise ValueError(f"Mel 滤波器形状应为 {(n_mels, 257)}，实际 {arrays['mel_filterbank'].shape}")
    if arrays["feature_39_after_cmvn"].shape[1] != 39:
        raise ValueError("最终特征维度必须为 39")


def _assert_finite(arrays: dict[str, np.ndarray]) -> None:
    for name, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
            raise ValueError(f"数组存在 NaN/Inf: {name}")
