from __future__ import annotations

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
    pre_emphasis,
)
from src.stft import compute_power_spectrum, compute_stft_per_frame


def test_pre_emphasis_length_and_constant() -> None:
    x = np.ones(8, dtype=np.float64)
    y = pre_emphasis(x, alpha=0.97)
    assert y.shape == x.shape
    assert np.isclose(y[0], 1.0)
    assert np.allclose(y[1:], 0.03)


def test_frame_signal_shapes() -> None:
    x = np.arange(1000, dtype=np.float64)
    frames, meta = frame_signal(x, frame_length=400, frame_step=160)
    assert frames.shape[1] == 400
    assert meta["frame_step"] == 160
    assert frames.shape[0] == meta["num_frames"]


def test_hamming_window_formula() -> None:
    win = build_hamming_window(10)
    assert win.shape == (10,)
    n = np.arange(10, dtype=np.float64)
    ref = 0.54 - 0.46 * np.cos((2.0 * np.pi * n) / 9.0)
    assert np.allclose(win, ref)


def test_stft_matches_unscaled_dft() -> None:
    rng = np.random.default_rng(0)
    frames = rng.standard_normal((3, 400))
    spectra = compute_stft_per_frame(frames, n_fft=512)
    assert spectra.shape == (3, 257)
    # np.fft.rfft 仅用于验证 SciPy STFT 标度换算，不参与主流水线
    assert np.allclose(spectra, np.fft.rfft(frames, n=512, axis=1))


def test_reference_alignment_frame_count() -> None:
    from src.reference_mfcc import build_aligned_reference_input

    signal = np.ones(52000)
    frames, meta = frame_signal(signal, 400, 160)
    y = build_aligned_reference_input(signal, 400, 160, 512, meta["num_frames"])
    librosa_frames = 1 + (y.size - 512) // 160
    assert librosa_frames == frames.shape[0]
    # librosa 第 t 帧的 400 点窗起点 = t*160 + 56，对应自定义帧起点 t*160
    t = 5
    assert np.allclose(y[t * 160 + 56 : t * 160 + 456], frames[t])


def test_mel_filterbank_shape_nonnegative() -> None:
    fb, _ = build_mel_filterbank(16000, 512, 26, 300.0, 8000.0)
    assert fb.shape == (26, 257)
    assert np.all(fb >= 0)


def test_end_to_end_core_shapes() -> None:
    cfg = MFCCConfig(project_root=Path(".").resolve())
    sr = cfg.sample_rate
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = 0.2 * np.sin(2 * np.pi * 440 * t)

    emphasized = pre_emphasis(signal, cfg.pre_emphasis_alpha)
    frames, _ = frame_signal(emphasized, cfg.frame_length, cfg.frame_step)
    win = build_hamming_window(cfg.frame_length)
    wframes = apply_window(frames, win)
    stft_matrix = compute_stft_per_frame(wframes, cfg.n_fft)
    power = compute_power_spectrum(stft_matrix)
    assert power.shape[1] == 257

    fb, _ = build_mel_filterbank(sr, cfg.n_fft, cfg.n_mels, cfg.mel_fmin, cfg.mel_fmax)
    mel = apply_mel_filterbank(power, fb)
    assert mel.shape[1] == 26
    assert np.all(np.isfinite(mel))

    log_mel = compute_log_mel(mel, cfg.log_epsilon)
    dct_full = compute_dct(log_mel)
    static_mfcc = extract_static_mfcc(dct_full, cfg.n_mfcc)
    log_energy = compute_log_energy(wframes, cfg.log_epsilon)
    static = assemble_static_feature(static_mfcc, log_energy)
    delta = compute_delta(static, window=cfg.delta_window)
    delta_delta = compute_delta_delta(static, window=cfg.delta_window)
    feat = assemble_final_feature(static, delta, delta_delta)
    feat_cmvn, mean, std = cmvn(feat, cfg.cmvn_epsilon)

    assert static.shape[1] == 13
    assert delta.shape == static.shape
    assert delta_delta.shape == static.shape
    assert feat.shape[1] == 39
    assert feat_cmvn.shape == feat.shape
    assert np.all(np.isfinite(feat_cmvn))
    # 非恒定维度应接近零均值和单位方差
    non_const = std[0] > 1e-6
    assert np.allclose(feat_cmvn[:, non_const].mean(axis=0), 0.0, atol=1e-5)
