from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_waveform(signal: np.ndarray, sr: int, path: Path, title: str) -> None:
    t = np.arange(signal.size) / sr
    plt.figure(figsize=(10, 3))
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    _save_fig(path)


def plot_two_waveforms(
    x1: np.ndarray, x2: np.ndarray, sr: int, path: Path, title: str, label1: str, label2: str
) -> None:
    t = np.arange(x1.size) / sr
    plt.figure(figsize=(10, 4))
    plt.plot(t, x1, label=label1, alpha=0.75)
    plt.plot(t, x2, label=label2, alpha=0.75)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    _save_fig(path)


def plot_hamming(window: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(7, 3))
    plt.plot(window)
    plt.title("Hamming Window")
    plt.xlabel("Sample Index")
    plt.ylabel("Window Value")
    _save_fig(path)


def plot_first_frame_compare(before: np.ndarray, after: np.ndarray, sr: int, path: Path) -> None:
    plt.figure(figsize=(9, 3))
    t_ms = np.arange(before.shape[0]) * 1000.0 / sr
    plt.plot(t_ms, before, label="Before Window", alpha=0.7)
    plt.plot(t_ms, after, label="After Window", alpha=0.7)
    plt.title("First Frame Before/After Windowing")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.legend()
    _save_fig(path)


def plot_spectrogram(
    mat: np.ndarray,
    path: Path,
    title: str,
    ylabel: str,
    frame_step: int | None = None,
    sample_rate: int | None = None,
    freq_hz: np.ndarray | None = None,
) -> None:
    plt.figure(figsize=(10, 4))
    kwargs = {"origin": "lower", "aspect": "auto", "interpolation": "nearest"}
    if frame_step is not None and sample_rate is not None:
        time_sec = mat.shape[0] * frame_step / sample_rate
        if freq_hz is not None:
            kwargs["extent"] = [0.0, time_sec, float(freq_hz[0]), float(freq_hz[-1])]
        else:
            kwargs["extent"] = [0.0, time_sec, 0.0, float(mat.shape[1])]
    plt.imshow(mat.T, **kwargs)
    plt.title(title)
    plt.xlabel("Time (s)" if frame_step is not None and sample_rate is not None else "Frame Index")
    plt.ylabel(ylabel)
    plt.colorbar(label="Value")
    _save_fig(path)


def plot_mel_filterbank(filterbank: np.ndarray, fft_freq_hz: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(10, 4))
    for i in range(filterbank.shape[0]):
        plt.plot(fft_freq_hz, filterbank[i], linewidth=1.0)
    plt.title("Mel Filterbank Responses")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain")
    _save_fig(path)


def plot_energy_curve(log_energy: np.ndarray, frame_step: int, sample_rate: int, path: Path) -> None:
    t = np.arange(log_energy.shape[0]) * frame_step / sample_rate
    plt.figure(figsize=(9, 3))
    plt.plot(t, log_energy[:, 0], label="Log Energy")
    plt.title("Log Short-Time Energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Log Energy")
    plt.legend()
    _save_fig(path)


def plot_three_energy_curves(
    e: np.ndarray,
    de: np.ndarray,
    dde: np.ndarray,
    frame_step: int,
    sample_rate: int,
    path: Path,
) -> None:
    t = np.arange(e.shape[0]) * frame_step / sample_rate
    plt.figure(figsize=(9, 4))
    plt.plot(t, e[:, 0], label="Static Energy")
    plt.plot(t, de[:, 0], label="Delta Energy")
    plt.plot(t, dde[:, 0], label="Delta-Delta Energy")
    plt.title("Static/Delta/Delta-Delta Energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()
    _save_fig(path)


def plot_mfcc_comparison(
    custom_mfcc: np.ndarray,
    ref_mfcc: np.ndarray,
    frame_step: int,
    sample_rate: int,
    path: Path,
) -> None:
    time_sec = custom_mfcc.shape[0] * frame_step / sample_rate
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(
        custom_mfcc.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[0.0, time_sec, 1.0, float(custom_mfcc.shape[1])],
    )
    plt.title("Custom MFCC")
    plt.xlabel("Time (s)")
    plt.ylabel("Coeff")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(
        ref_mfcc.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[0.0, time_sec, 1.0, float(ref_mfcc.shape[1])],
    )
    plt.title("Reference MFCC")
    plt.xlabel("Time (s)")
    plt.ylabel("Coeff")
    plt.colorbar()
    _save_fig(path)


def plot_error_heatmap(
    custom_mfcc: np.ndarray,
    ref_mfcc: np.ndarray,
    frame_step: int,
    sample_rate: int,
    path: Path,
) -> None:
    err = custom_mfcc - ref_mfcc
    time_sec = custom_mfcc.shape[0] * frame_step / sample_rate
    plt.figure(figsize=(10, 4))
    plt.imshow(
        err.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        extent=[0.0, time_sec, 1.0, float(err.shape[1])],
    )
    plt.title("MFCC Error Heatmap (Custom - Reference)")
    plt.xlabel("Time (s)")
    plt.ylabel("Coeff")
    plt.colorbar(label="Error")
    _save_fig(path)
